import re
import time
import pickle
import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Classical NLP: Regex, Edit Distance, and ELIZA Rules (Area B)
KEYWORDS = {
    "account", "balance", "transfer", "credit", "card", "payment",
    "flight", "hotel", "booking", "refund", "cancel", "password", "device"
}

def edit_distance(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]

def fix_spelling(text):
    words, fixed = [], []
    for w in text.split():
        if w.isalpha() and len(w) >= 4 and w not in KEYWORDS:
            match = next((k for k in KEYWORDS if abs(len(w) - len(k)) <= 1 and edit_distance(w, k) == 1), w)
            if match != w:
                fixed.append(f"{w} -> {match}")
            words.append(match)
        else:
            words.append(w)
    return " ".join(words), fixed

def preprocess(raw_text):
    codes = re.findall(r'\b[A-Za-z]{2,4}-\d{3,4}\b', raw_text)
    text = raw_text.lower().strip()
    text = re.sub(r'\b[a-z]{2,4}-\d{3,4}\b', ' model_code ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned, fixed = fix_spelling(text)
    return cleaned, codes, fixed

SMALL_TALK = [
    (r'^(hi|hello|hey)\b', "Hello! How can I help route your support request today?"),
    (r'\bhow are you\b', "I'm doing well! Tell me what issue you need help with."),
    (r'\b(thanks|thank you)\b', "You're welcome! Let me know if you need anything else."),
    (r'^(bye|goodbye)\b', "Goodbye!")
]

def check_small_talk(text):
    for pattern, reply in SMALL_TALK:
        if re.search(pattern, text.lower().strip()):
            return reply
    return None

def assign_team(intent, conf):
    if intent == "oos" or conf < 0.45:
        return "Human Agent (Low Confidence / Out-of-Scope)"
    if any(k in intent for k in ["card", "balance", "transfer", "pay", "bill", "bank", "loan", "credit"]):
        return "Billing & Finance Team"
    if any(k in intent for k in ["flight", "hotel", "travel", "book", "car", "uber"]):
        return "Travel & Logistics Team"
    if any(k in intent for k in ["phone", "sync", "device", "password", "pin", "update"]):
        return "Technical Support Team"
    return "General Support Team"

# 2. Load the 4 Trained Models
with open("saved_models/model1_tfidf_lr.pkl", "rb") as f:
    m1 = pickle.load(f)
tfidf, clf_lr, label_names = m1["tfidf"], m1["model"], m1["labels"]

class AveragedEmbeddingFFNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.fc(self.embedding(x))

m2_ckpt = torch.load("saved_models/model2_ffnn.pt", map_location="cpu")
vocab, MAX_LEN = m2_ckpt["vocab"], m2_ckpt["max_len"]
nn_model = AveragedEmbeddingFFNN(len(vocab), 128, 256, len(label_names))
nn_model.load_state_dict(m2_ckpt["state_dict"])
nn_model.eval()

def encode_words(text):
    ids = [vocab.get(w, 1) for w in text.split()[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

tokenizer = AutoTokenizer.from_pretrained("saved_models/model3_bert")
bert_fp32 = AutoModelForSequenceClassification.from_pretrained("saved_models/model3_bert").to("cpu").eval()

bert_q_base = AutoModelForSequenceClassification.from_pretrained("saved_models/model3_bert").to("cpu").eval()
quantized_bert = torch.ao.quantization.quantize_dynamic(bert_q_base, {torch.nn.Linear}, dtype=torch.qint8)
quantized_bert.load_state_dict(torch.load("saved_models/model4_quantized_int8.pt", map_location="cpu"))
quantized_bert.eval()

# 3. Prediction Function
def predict(message):
    if not message or not message.strip():
        return "Please enter a customer message.", "—", "None", "None", []

    eliza_reply = check_small_talk(message)
    if eliza_reply:
        return "ELIZA Rule Engine (No ML Needed)", eliza_reply, "None", "None", []

    cleaned, codes, fixed = preprocess(message)

    # Model 1: TF-IDF + Logistic Regression
    t0 = time.perf_counter()
    p1 = clf_lr.predict_proba(tfidf.transform([cleaned]))[0]
    i1, c1, lat1 = int(p1.argmax()), float(p1.max()), (time.perf_counter() - t0) * 1000

    # Model 2: Feedforward NN on Word Embeddings
    t0 = time.perf_counter()
    with torch.no_grad():
        p2 = torch.softmax(nn_model(torch.tensor([encode_words(cleaned)], dtype=torch.long)), dim=-1)[0]
    i2, c2, lat2 = int(p2.argmax()), float(p2.max()), (time.perf_counter() - t0) * 1000

    # Model 3: Fine-tuned BERT-Tiny (FP32)
    t0 = time.perf_counter()
    inp = tokenizer(cleaned, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        p3 = torch.softmax(bert_fp32(**inp).logits, dim=-1)[0]
    i3, c3, lat3 = int(p3.argmax()), float(p3.max()), (time.perf_counter() - t0) * 1000

    # Model 4: Quantized BERT-Tiny (INT8 - Primary Model)
    t0 = time.perf_counter()
    with torch.no_grad():
        p4 = torch.softmax(quantized_bert(**inp).logits, dim=-1)[0]
    i4, c4, lat4 = int(p4.argmax()), float(p4.max()), (time.perf_counter() - t0) * 1000

    intent = label_names[i4]
    team = assign_team(intent, c4)
    intent_display = f"{intent} ({c4*100:.1f}% confidence)"
    codes_display = ", ".join(codes) if codes else "None"
    fixes_display = ", ".join(fixed) if fixed else "None"

    table = [
        ["1. TF-IDF + Logistic Regression", label_names[i1], f"{c1*100:.1f}%", f"{lat1:.2f} ms"],
        ["2. Word Embeddings + FFNN (SGD)", label_names[i2], f"{c2*100:.1f}%", f"{lat2:.2f} ms"],
        ["3. BERT-Tiny (FP32)", label_names[i3], f"{c3*100:.1f}%", f"{lat3:.2f} ms"],
        ["4. Quantized BERT-Tiny (INT8)", intent, f"{c4*100:.1f}%", f"{lat4:.2f} ms"],
    ]
    return team, intent_display, codes_display, fixes_display, table

# 4. Clean Gradio UI (Hides footer links)
custom_css = "footer {display: none !important;}"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Customer Support Ticket Router") as demo:
    gr.Markdown("## Customer Support Ticket Router")
    gr.Markdown("Automated ticket routing powered by Classical NLP preprocessing and a Quantized INT8 Transformer.")

    with gr.Row():
        with gr.Column(scale=1):
            msg_input = gr.Textbox(
                label="Customer Message",
                placeholder="Type a support message (e.g., check my acount balnce for card DW-6534)...",
                lines=3
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Route Ticket", variant="primary")

            gr.Examples(
                examples=[
                    ["hello there!"],
                    ["check my acount balnce for card DW-6534"],
                    ["i need to book a flight to istanbul tomorrow"],
                    ["what is the recipe for chocolate cake?"]
                ],
                inputs=msg_input
            )

        with gr.Column(scale=1):
            team_out = gr.Textbox(label="Assigned Team", interactive=False)
            intent_out = gr.Textbox(label="Primary Model Prediction (Quantized INT8)", interactive=False)
            with gr.Row():
                codes_out = gr.Textbox(label="Extracted Model Codes", interactive=False)
                fixes_out = gr.Textbox(label="Spelling Fixes", interactive=False)

    table_out = gr.Dataframe(
        headers=["Model", "Predicted Intent", "Confidence", "Latency"],
        label="Side-by-Side Model Comparison",
        interactive=False
    )

    submit_btn.click(
        fn=predict,
        inputs=msg_input,
        outputs=[team_out, intent_out, codes_out, fixes_out, table_out]
    )
    msg_input.submit(
        fn=predict,
        inputs=msg_input,
        outputs=[team_out, intent_out, codes_out, fixes_out, table_out]
    )
    clear_btn.click(
        fn=lambda: ("", "", "", "", "", []),
        outputs=[msg_input, team_out, intent_out, codes_out, fixes_out, table_out]
    )

if __name__ == "__main__":
    demo.launch()
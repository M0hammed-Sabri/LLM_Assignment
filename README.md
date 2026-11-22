# Customer Support Ticket Router (SWE015)

An end-to-end intelligent customer support ticket routing application comparing four models of increasing sophistication on the CLINC150 (`clinc_oos` plus) dataset:
1. **Classical Baseline:** TF-IDF + Logistic Regression
2. **Neural Network:** Averaged Word Embeddings + Feedforward Neural Network (trained with SGD)
3. **Transformer:** Fine-tuned `google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny FP32)
4. **Quantized Transformer:** Dynamic INT8 Quantized BERT-Tiny (primary runtime model)

## Setup & Run

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone https://github.com/M0hammed-Sabri/LLM_Assignment.git
   cd <YOUR_REPO_FOLDER>
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate

1.Install dependencies:

    pip install -r requirements.txt

2.Download pre-trained model weights (or re-train via train.ipynb):

    Download saved_models.zip from Google Drive: <https://drive.google.com/file/d/17nx98-W7AJrysd47C45zJcQiYYW-_EFE/view?usp=sharing>
    Extract saved_models/ into the project root directory.

3.Run the application (single run command):

    python app.py

    Open http://127.0.0.1:7860 in your browser.
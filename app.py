# --- 1. Standard Library Imports ---
import os
import json
import traceback

# --- 2. Third-Party Library Imports: Scientific Computing & ML ---
import numpy as np
import joblib
import torch
from typing import Tuple, Optional
# --- 3. Third-Party Library Imports: Web Framework & Tools ---
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# --- 4. Third-Party Library Imports: Hugging Face Transformers ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 2. Basic Configuration (Basic Config) ---
app = Flask(__name__)

# Configure Cross-Origin Resource Sharing (CORS)
# 1. General Configuration for all paths (/*): Allow frontend origin access
CORS(app, resources={r"/*": {"origins": "https://akwve.github.io"}})
CORS(app, resources={
    r"/predict*": {
        "origins": "https://akwve.github.io",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Frontend Domain (GitHub Pages)
FRONTEND_ORIGIN = "https://akwve.github.io"


# --- 3. Model & Vectorizer Path Configuration (Paths & Device) ---

# Model and Vectorizer Paths (Ensure these match paths on AWS/server)
MODEL_PATH  = "/home/ubuntu/fake_news_api/bert_fake_news_model"
TFIDF_PATH  = "/home/ubuntu/fake_news_api/tfidf_vectorizer.pkl"
XGB_PATH    = "/home/ubuntu/fake_news_api/xgboost_fake_news_model.pkl"

USE_CUDA = False
device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"

# --- 4. Model Loading and Initialization ---
# -------------------- Load BERT --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("✅ Loaded BERT from:", MODEL_PATH, "| id2label:", model.config.id2label)

BERT_LABEL_MAP = {0: "False", 1: "Real"}
XGB_LABEL_MAP = {0: "Fake", 1: "Real"}
# -------------------- Load XGB + TF-IDF --------------------
tfidf = joblib.load(TFIDF_PATH)
xgb_model = joblib.load(XGB_PATH)
print("✅ Loaded TFIDF:", TFIDF_PATH)
print("✅ Loaded XGBoost:", XGB_PATH)

# --- 5.OpenAI Client Initialization ---
# -------------------- OpenAI (GPT-4o) setup --------------------
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"

GPT_MODEL = "gpt-4o-mini"

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI client initialized (GPT-4o-mini enabled)")
except Exception as e:
    client = None
    print("⚠️ Failed to initialize OpenAI client:", e)

# --- 6. ChatGPT Utility Function (For Prediction Enhancement) ---
def call_gpt_label(text: str):
    """ Call GPT-4o-mini to evaluate whether a tweet is true or false.
    Returns a string like 'True, confidence 0.97' or None on failure."""
    if client is None:
        print("⚠️ ChatGPT client not initialized. Skipping GPT call.")
        return None

    prompt = (
        "Determine if the following tweet contains true or false information. "
        "Answer STRICTLY in one of two formats: 'True, confidence X.XX' or 'False, confidence X.XX'. "
        "The confidence score (X.XX) must be between 0.00 and 1.00."
        "Answer strictly in this format: 'True, confidence 0.97' or 'False, confidence 0.25'. "
        f"Tweet: {text}"
    )

    try:
        resp = client.chat.completions.create(
      s      model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        # Extract the content and strip whitespace
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("ChatGPT error:", e)
        return None

# --- 7. CORS Response Header Global Handler ---
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"]  = FRONTEND_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "OPTIONS, POST"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Max-Age"]       = "3600"
    return resp

# -------------------- Utilities --------------------


def run_bert(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf_tensor = probs[0][pred]
            conf = conf_tensor.item() 
    except Exception as e:
        print(f"BERT error: {e}")
        return "Error", 0.0

    return BERT_LABEL_MAP.get(pred, "Unknown"), conf

def run_xgb(text: str) -> Tuple[str, float]:
   
    if tfidf is None or xgb_model is None:
        print("❌ XGBoost model or vectorizer not loaded. Cannot run prediction.")
        return "Error", 0.0
    try:
        
        X = tfidf.transform([text]).astype(np.float32) 
        
        proba = xgb_model.predict_proba(X)[0] 
        
        # Prediction Index and Confidence
        idx = int(np.argmax(proba)) 
        conf = float(proba[idx])

    except Exception as e:
        print(f"❌ XGBoost prediction error: {e}")
        return "Error", 0.0
    
    label = XGB_LABEL_MAP.get(idx, "Unknown")
    return label, conf

def call_gpt_label(text: str) -> Optional[str]:
    if client is None:
        return None


    SYSTEM_PROMPT = "Determine if the following text contains true or false information. Answer STRICTLY in this >
    USER_PROMPT = f"Tweet: {text}" 

    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            temperature=0.1, 
            max_tokens=50 
        )
        return resp.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ ChatGPT API error: {e}")
        return None
    
# -------------------- Routes --------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict_xgb", methods=["POST", "OPTIONS"])
def predict_xgb():
    if request.method == "OPTIONS":
        return make_response("", 204) 

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    
    xgb_label, xgb_conf = "N/A", None
    gpt_label = "N/A" 

    try:
        # 1. XGBoost Prediction
        xgb_label, xgb_conf = run_xgb(text) 
        
        # 2. ChatGPT Prediction
        gpt_result = call_gpt_label(text)
        gpt_label = gpt_result if gpt_result else "Disabled/Failed"
        
        # 3. return
        return jsonify({
            "xgb_label": xgb_label, 
            "xgb_confidence": round(xgb_conf, 3) if xgb_conf is not None else None,
            "gpt_label": gpt_label
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"XGB/GPT error: {str(e)}"}), 500
    
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response= make_response("", 204)
        response.headers["Access-Control-Allow-Origin"] = "https://akwve.github.io"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response

    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # 1. BERT
        bert_label, bert_conf = run_bert(text)


        # 3. ChatGPT
        gpt_label = call_gpt_label(text) or "N/A"

        # 4. return
        return jsonify({
            "bert_label": bert_label,
            "bert_confidence": round(bert_conf, 3),
            "gpt_label": gpt_label
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
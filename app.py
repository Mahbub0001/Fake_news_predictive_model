import os
import pickle
from typing import Tuple, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gradio as gr


MODEL_PATH = "model.pkl"
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"


def _load_text_column(df: pd.DataFrame) -> pd.Series:
    # prefer common name 'text', otherwise use first column
    for col in ["text", "Text", "TEXT", "title", "content"]:
        if col in df.columns:
            return df[col].astype(str)
    # fallback to first column
    return df.iloc[:, 0].astype(str)


def load_model_and_vectorizer(model_path: str = MODEL_PATH) -> Tuple[object, TfidfVectorizer]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    texts = []
    for p in (FAKE_CSV, TRUE_CSV):
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                texts.append(_load_text_column(df))
            except Exception:
            
                continue

    if not texts:
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
        
        vectorizer.fit([""])
    else:
        corpus = pd.concat(texts, axis=0).astype(str).tolist()
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
        vectorizer.fit(corpus)

    return model, vectorizer


def predict_news(text: str, model_and_vect=None) -> Tuple[Dict[str, float], str]:
    if model_and_vect is None:
        # lazy-load on first prediction
        model, vectorizer = load_model_and_vectorizer()
    else:
        model, vectorizer = model_and_vect

    if not isinstance(text, str) or text.strip() == "":
        return {"Fake": 0.0, "True": 0.0}, "Please enter some text"

    X = vectorizer.transform([text])

    try:
        probs = model.predict_proba(X)
    
        p_true = float(probs[:, 1][0])
    except Exception:
    
        try:
            pred = model.predict(X)[0]
            p_true = 1.0 if pred == 1 else 0.0
        except Exception:
            p_true = 0.0

    p_true = max(0.0, min(1.0, p_true))
    p_fake = 1.0 - p_true

    label = "True" if p_true >= 0.5 else "Fake"
    probs_dict = {"Fake": float(p_fake), "True": float(p_true)}
    return probs_dict, f"{label} ({p_true*100:.2f}%)"


def build_demo():
    model_and_vect = None
    try:
        model_and_vect = load_model_and_vectorizer()
    except Exception as e:
        # We'll still start the UI but show errors on load
        model_and_vect = None

    title = "Fake News Detector"
    description = (
        "Enter a news text and the model will predict whether it's Fake or True. "
        "Confidence and class probabilities are shown."
    )

    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            inp = gr.Textbox(lines=6, placeholder="Paste news text here...", label="News Text")
        with gr.Row():
            lbl = gr.Label(label="Predicted probabilities")
            conf = gr.Textbox(label="Result (label and confidence)")

        def _predict(text: str):
            return predict_news(text, model_and_vect)

        btn = gr.Button("Predict")
        btn.click(fn=_predict, inputs=inp, outputs=[lbl, conf])

        gr.Examples(examples=[
            "The economy is booming and unemployment has fallen to zero.",
            "Celebrity endorses miracle cure that doctors refuse to acknowledge."
        ], inputs=inp)
        gr.Markdown("Made by Mahbub Ul Alam Bhuiyan")
    return demo



demo = build_demo()
demo.launch()

"""
Pre-training & Artifact Export Script for AI Resume Analyzer

Run this script once locally before deploying or launching app.py:
    python train_export.py

It processes the dataset and exports pre-trained models into the `./data/` folder.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Paths configuration matching app.py layout
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "resume_data (1).csv")
MODEL_EXPORT_PATH = os.path.join(DATA_DIR, "trained_pipeline.pkl")


def clean_resume_text(txt: str) -> str:
    """Regex-based text normalization matching app.py cleaning logic."""
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+|https\S+", " ", txt)
    txt = re.sub(r"@\S+", " ", txt)
    txt = re.sub(r"#\S+", " ", txt)
    txt = re.sub(r"\brt\b", " ", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s\-/._]", " ", txt)
    txt = re.sub(r"[^\x00-\x7f]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Target dataset '{CSV_PATH}' not found.")
        print("Please ensure 'resume_data (1).csv' is placed inside the 'data/' folder.")
        return

    print("Loading dataset from:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Clean dataset and drop missing/duplicate entries
    df = df.dropna(subset=["Category", "Resume"]).reset_index(drop=True)
    df["cleaned"] = df["Resume"].apply(clean_resume_text)
    df = df.drop_duplicates(subset="cleaned").reset_index(drop=True)

    print(f"Dataset successfully loaded. Total samples: {len(df)}")

    # Encode category labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize text using TF-IDF (Matching parameters in app.py)
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        stop_words="english", max_features=5000, ngram_range=(1, 2), sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Train Logistic Regression Model
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Test Accuracy: {test_accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Pre-calculate category centroids for cosine similarity comparisons
    print("Calculating category centroids...")
    X_full_vec = tfidf.transform(df["cleaned"])
    centroids = {}
    for idx, cls_name in enumerate(label_encoder.classes_):
        mask = y == idx
        centroids[cls_name] = np.asarray(X_full_vec[mask].mean(axis=0))

    # Package pipeline payload
    pipeline_payload = {
        "tfidf": tfidf,
        "model": model,
        "label_encoder": label_encoder,
        "centroids": centroids,
        "test_accuracy": test_accuracy,
        "n_samples": len(df),
        "n_categories": len(label_encoder.classes_),
    }

    # Save artifact payload
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_EXPORT_PATH, "wb") as f:
        pickle.dump(pipeline_payload, f)

    print(f"Artifact successfully saved to: {MODEL_EXPORT_PATH}\n")


if __name__ == "__main__":
    main()
"""
train_model.py
AI-Based Scam Message Detection System — JKUAT IT Department
Trains a Logistic Regression classifier on the combined dataset
(SMS Spam Collection + EduPhish + JKUAT custom messages),
evaluates it, and saves the model and vectorizer as .pkl files.
Run once before starting app.py.
"""

import re
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATASET_PATH    = BASE_DIR / "combined_dataset.csv"   # merged dataset
MODEL_DIR       = BASE_DIR / "model"
MODEL_PATH      = MODEL_DIR / "scam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_DIR.mkdir(exist_ok=True)


# ── 1. Load combined dataset ───────────────────────────────────────────────────
print("Loading combined dataset...")
df = pd.read_csv(DATASET_PATH, encoding="utf-8")

# Expected columns: label (scam/legitimate), message, source
df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["scam", "legitimate"])]

print(f"  Total samples : {len(df):,}")
print(f"  Scam          : {(df['label'] == 'scam').sum():,}")
print(f"  Legitimate    : {(df['label'] == 'legitimate').sum():,}")
if "source" in df.columns:
    print("\n  Breakdown by source:")
    for src, grp in df.groupby("source"):
        print(f"    {src:<28} {len(grp):>6,} rows")


# ── 2. NLP preprocessing ───────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Mirrors the preprocess_text() in app.py exactly.
    Lowercases, replaces URLs with 'link', removes non-alphanumeric
    characters, and normalises whitespace.
    """
    text = str(text).lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", " link ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\nPreprocessing messages...")
df["clean_message"] = df["message"].apply(preprocess_text)


# ── 3. TF-IDF feature extraction ───────────────────────────────────────────────
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=8000,      # larger vocab for the bigger combined dataset
    ngram_range=(1, 2),     # unigrams and bigrams
    sublinear_tf=True,      # log normalisation on term frequencies
    min_df=2                # ignore terms that appear in fewer than 2 docs
)
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]


# ── 4. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples : {X_train.shape[0]:,}")
print(f"  Test samples     : {X_test.shape[0]:,}")


# ── 5. Train Logistic Regression ───────────────────────────────────────────────
print("\nTraining Logistic Regression classifier...")
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",   # handles class imbalance
    C=1.0
)
model.fit(X_train, y_train)


# ── 6. Evaluate ────────────────────────────────────────────────────────────────
print("\n── Evaluation Results ──────────────────────────────────────────────")
y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="scam")
recall    = recall_score(y_test, y_pred, pos_label="scam")
f1        = f1_score(y_test, y_pred, pos_label="scam")

print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-score  : {f1:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["legitimate", "scam"]))
print("────────────────────────────────────────────────────────────────────")


# ── 7. Save model and vectorizer ───────────────────────────────────────────────
print(f"\nSaving model to      {MODEL_PATH}")
print(f"Saving vectorizer to {VECTORIZER_PATH}")
joblib.dump(model,      MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print("\nDone. You can now start app.py.")
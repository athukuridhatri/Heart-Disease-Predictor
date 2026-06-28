"""
train_model.py
==============
Training script for the Heart Disease Risk Predictor.
Uses the UCI Cleveland Heart Disease Dataset (278 clean samples, 13 features).

Run once to generate model.pkl and evaluation plots:
    python train_model.py
"""

import os
import pickle

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH  = "heart.csv"
MODEL_PATH    = "model.pkl"
PLOTS_DIR     = os.path.join("static", "plots")
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Dataset (UCI Cleveland Heart Disease — real data)
# ---------------------------------------------------------------------------
print("=" * 55)
print("  Heart Disease Predictor — Model Training")
print("=" * 55)
print(f"\n[1/5] Loading dataset from '{DATASET_PATH}' ...")

df = pd.read_csv(DATASET_PATH)

# Ensure target is binary: 0 = no disease, 1 = disease
if df["target"].max() > 1:
    df["target"] = (df["target"] > 0).astype(int)

print(f"      Rows: {len(df)} | Features: {df.shape[1] - 1}")
print(f"      Target distribution:\n{df['target'].value_counts().rename({0: 'No Disease (0)', 1: 'Disease (1)'}).to_string()}")

# ---------------------------------------------------------------------------
# 2. Prepare Features
# ---------------------------------------------------------------------------
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

X = df[FEATURES]
y = df["target"]

# 80 / 20 split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[2/5] Train / Test split (80 / 20, stratified)")
print(f"      Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# Standardise features (mean=0, std=1) — critical for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Train Logistic Regression
# ---------------------------------------------------------------------------
print("\n[3/5] Training Logistic Regression model ...")
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="lbfgs")
model.fit(X_train_sc, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate Model
# ---------------------------------------------------------------------------
print("\n[4/5] Evaluating model ...")

y_pred      = model.predict(X_test_sc)
y_prob      = model.predict_proba(X_test_sc)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)
cm        = confusion_matrix(y_test, y_pred)
report    = classification_report(y_test, y_pred, output_dict=True)

print(f"\n  {'Metric':<20} {'Score':>8}")
print(f"  {'-'*30}")
print(f"  {'Accuracy':<20} {accuracy:>8.4f}  ({accuracy:.2%})")
print(f"  {'Precision':<20} {precision:>8.4f}")
print(f"  {'Recall':<20} {recall:>8.4f}")
print(f"  {'F1 Score':<20} {f1:>8.4f}")
print(f"  {'ROC-AUC':<20} {roc_auc:>8.4f}")
print(f"\n  Confusion Matrix:\n{cm}")

# ---------------------------------------------------------------------------
# 4a. Plot: Confusion Matrix
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Reds",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — Logistic Regression", fontsize=12, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Confusion matrix saved → {cm_path}")

# ---------------------------------------------------------------------------
# 4b. Plot: ROC Curve
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="#e63946", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#555", lw=1.2, linestyle="--", label="Random Classifier")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Logistic Regression", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  ROC curve saved          → {roc_path}")

# ---------------------------------------------------------------------------
# 5. Save Model Artefact
# ---------------------------------------------------------------------------
print(f"\n[5/5] Saving model to '{MODEL_PATH}' ...")

model_data = {
    "model":            model,
    "scaler":           scaler,
    "features":         FEATURES,
    # Evaluation metrics
    "accuracy":         accuracy,
    "precision":        precision,
    "recall":           recall,
    "f1_score":         f1,
    "roc_auc":          roc_auc,
    "confusion_matrix": cm.tolist(),
    "report":           report,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'='*55}")
print(f"  ✅ Training complete!")
print(f"     Accuracy  : {accuracy:.2%}")
print(f"     Precision : {precision:.4f}")
print(f"     Recall    : {recall:.4f}")
print(f"     F1 Score  : {f1:.4f}")
print(f"     ROC-AUC   : {roc_auc:.4f}")
print(f"{'='*55}")
print(f"\n  → Run: python app.py")

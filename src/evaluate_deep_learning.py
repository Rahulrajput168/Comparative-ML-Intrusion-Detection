import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# Assumes this script is inside the project's src/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def read_csv_checked(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)

X_test = read_csv_checked("X_test.csv")
y_test = read_csv_checked("y_test.csv").squeeze()
model_path = os.path.join(MODEL_DIR, "deep_learning_mlp.pkl")
if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}. Run train_deep_learning.py first."
    )

model = joblib.load(model_path)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("=" * 60)
print("DEEP LEARNING MLP EVALUATION")
print("=" * 60)
print(f"Accuracy  : {accuracy * 100:.4f}%")
print(f"Precision : {precision * 100:.4f}%")
print(f"Recall    : {recall * 100:.4f}%")
print(f"F1 Score  : {f1 * 100:.4f}%")
print("\nClassification report:")
print(classification_report(
    y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Normal", "Attack"]
)
display.plot()
plt.title("Deep Learning MLP Confusion Matrix")
plt.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_deep_learning.png")
plt.savefig(cm_path, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nConfusion matrix saved: {cm_path}")

results = pd.DataFrame([{
    "Model": "Deep Learning - MLP",
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100
}])
results_path = os.path.join(REPORT_DIR, "deep_learning_evaluation.csv")
results.to_csv(results_path, index=False)
print(f"Evaluation results saved: {results_path}")

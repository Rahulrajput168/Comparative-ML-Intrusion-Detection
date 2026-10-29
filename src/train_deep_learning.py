
import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# Assumes this script is inside the project's src/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 68)
print("DEEP LEARNING MLP - SCALED TRAINING AND VALIDATION TUNING")
print("=" * 68)

def read_csv_checked(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)
X_train = read_csv_checked("X_train.csv")
y_train = read_csv_checked("y_train.csv").squeeze()
X_test = read_csv_checked("X_test.csv")
y_test = read_csv_checked("y_test.csv").squeeze()

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        f"Feature mismatch: train has {X_train.shape[1]} columns, "
        f"test has {X_test.shape[1]} columns."
    )
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("X_train.csv and X_test.csv columns/order do not match.")
if X_train.isna().any().any() or X_test.isna().any().any():
    raise ValueError("Missing values found. Apply the same imputation in preprocessing first.")

print(f"X_train: {X_train.shape}; X_test: {X_test.shape}")
print("Using StandardScaler inside a Pipeline to prevent preprocessing leakage.")

# Keep the held-out test set out of model selection.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# A small, practical search. Accuracy is selected on validation data only.
candidates = [
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (256, 128, 64), "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (256, 128, 64), "alpha": 0.001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (128, 64), "alpha": 0.0001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (192, 96, 48), "alpha": 0.0005, "learning_rate_init": 0.001},
]

best_params = None
best_val_accuracy = -1.0
start_all = time.time()

for i, params in enumerate(candidates, start=1):
    print("\n" + "-" * 68)
    print(f"Candidate {i}/{len(candidates)}: {params}")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=params["alpha"],
            batch_size=256,
            learning_rate_init=params["learning_rate_init"],
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=42,
            verbose=False
        ))
    ])
    pipeline.fit(X_fit, y_fit)
    val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_acc * 100:.3f}%")
    print(f"Training iterations: {pipeline.named_steps['mlp'].n_iter_}")

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_params = params

print("\n" + "=" * 68)
print("BEST VALIDATION CONFIGURATION")
print(best_params)
print(f"Validation accuracy: {best_val_accuracy * 100:.3f}%")
print("Retraining the selected configuration on the full training set...")

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=best_params["alpha"],
        batch_size=256,
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=150,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=42,
        verbose=True
    ))
])

start = time.time()
final_model.fit(X_train, y_train)
training_time = time.time() - start

# One final evaluation on the held-out test set.
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 68)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 68)
print(f"Accuracy  : {accuracy * 100:.4f}%")
print(f"Precision : {precision * 100:.4f}%")
print(f"Recall    : {recall * 100:.4f}%")
print(f"F1 Score  : {f1 * 100:.4f}%")
print(f"Training time: {training_time:.2f} seconds")
print("\nClassification report:")
print(classification_report(
    y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0
))

model_path = os.path.join(MODEL_DIR, "deep_learning_mlp.pkl")
joblib.dump(final_model, model_path)
print(f"\nUpdated model saved to: {model_path}")

results = pd.DataFrame([{
    "Model": "Deep Learning - MLP",
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100,
    "Training Time": training_time,
    "Best Hidden Layers": str(best_params["hidden_layer_sizes"]),
    "Best Alpha": best_params["alpha"],
    "Best Learning Rate": best_params["learning_rate_init"],
    "Validation Accuracy": best_val_accuracy * 100
}])

for filename in ["deep_learning_results.csv", "deep_learning_evaluation.csv"]:
    out_path = os.path.join(REPORT_DIR, filename)
    results.to_csv(out_path, index=False)
    print(f"Metrics saved to: {out_path}")

print("\nDone. Use the printed test accuracy as the actual result; do not manually change metrics.")


import os
import sys
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.transformer_model import TabularTransformer

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "transformer_model.pt")

# Load the processed official test data
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).iloc[:, 0].to_numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = TabularTransformer(input_features=X_test.shape[1]).to(device)

# Support checkpoints saved either as a state dict or a dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

with torch.no_grad():
    inputs = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
    predictions = []

    for batch in inputs.split(512):
        outputs = model(batch)
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())

predictions = np.asarray(predictions)

# Load original test labels, retaining the multiclass attack names
from dataset.load_dataset import load_data
_, original_test = load_data()

actual_names = original_test["label"].astype(str).to_numpy()
actual_binary = np.where(actual_names == "normal", 0, 1)

if len(actual_binary) != len(predictions):
    raise ValueError("Prediction count and original test-label count do not match.")

results = pd.DataFrame({
    "actual_attack_type": actual_names,
    "actual_binary": actual_binary,
    "predicted_binary": predictions
})

results["correct"] = results["actual_binary"] == results["predicted_binary"]

print("\nOverall accuracy:", round(results["correct"].mean() * 100, 2), "%")

print("\nPerformance by actual label:")
summary = results.groupby("actual_attack_type").agg(
    samples=("correct", "size"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean")
)
summary["accuracy_percent"] = (summary["accuracy"] * 100).round(2)
print(summary.drop(columns="accuracy").sort_values("samples", ascending=False).to_string())

print("\nAttack types most often predicted as Normal:")
missed = results[
    (results["actual_binary"] == 1) &
    (results["predicted_binary"] == 0)
]
print(missed["actual_attack_type"].value_counts().head(15).to_string())

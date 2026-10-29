import os
import sys
import torch
import pandas as pd

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from transformer_model import TabularTransformer


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "transformer_model.pt"
)

X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "processed_data",
    "X_test.csv"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "processed_data",
    "y_test.csv"
)


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------

print("Loading Transformer model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model = TabularTransformer(
    input_features=checkpoint["input_features"],
    d_model=checkpoint["d_model"],
    nhead=checkpoint["nhead"],
    num_layers=checkpoint["num_layers"],
    dim_feedforward=checkpoint["dim_feedforward"],
    dropout=checkpoint["dropout"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Transformer model loaded successfully.")


# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

print("\nLoading test data...")

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# SELECT ONE SAMPLE
# ------------------------------------------------------------

sample_number = 0

sample = X_test.iloc[sample_number]

actual_label = int(y_test[sample_number])


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

x = torch.tensor(
    sample.values,
    dtype=torch.float32
).unsqueeze(0)


with torch.no_grad():

    output = model(x)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    prediction = int(
        torch.argmax(
            probabilities
        ).item()
    )


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

labels = {
    0: "Normal Traffic",
    1: "Attack Traffic"
}


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRANSFORMER SINGLE-SAMPLE PREDICTION")
print("=" * 60)

print(f"Sample Number : {sample_number}")

print(
    f"Prediction    : {labels[prediction]}"
)

print(
    f"Actual Label  : {labels[actual_label]}"
)

print(
    f"Normal Prob.  : {probabilities[0].item() * 100:.2f}%"
)

print(
    f"Attack Prob.  : {probabilities[1].item() * 100:.2f}%"
)

print("=" * 60)
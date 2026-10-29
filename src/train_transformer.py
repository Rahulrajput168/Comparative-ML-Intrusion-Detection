import os
import sys
import random
import copy
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.transformer_model import TabularTransformer

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
VALIDATION_SIZE = 0.15
PATIENCE = 7

INPUT_FEATURES = 122
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 65)
print("TRANSFORMER-BASED INTRUSION DETECTION")
print("=" * 65)
print("Device:", DEVICE)

# --------------------------------------------------
# 3. LOAD PROCESSED DATA
# --------------------------------------------------

X_train_df = pd.read_csv(
    os.path.join(DATA_DIR, "X_train.csv")
)
X_test_df = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_train_df = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
)
y_test_df = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
)

y_train = y_train_df.iloc[:, 0].to_numpy(dtype=np.int64)
y_test = y_test_df.iloc[:, 0].to_numpy(dtype=np.int64)

print("\nDataset loaded successfully.")
print("Training features:", X_train_df.shape)
print("Testing features:", X_test_df.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

# --------------------------------------------------
# 4. DATA VALIDATION
# --------------------------------------------------

if X_train_df.shape[1] != INPUT_FEATURES:
    raise ValueError(
        f"Expected {INPUT_FEATURES} features, "
        f"but training data has {X_train_df.shape[1]}."
    )

if X_test_df.shape[1] != INPUT_FEATURES:
    raise ValueError(
        f"Expected {INPUT_FEATURES} features, "
        f"but testing data has {X_test_df.shape[1]}."
    )

if list(X_train_df.columns) != list(X_test_df.columns):
    raise ValueError(
        "Training and testing feature columns/order do not match."
    )

if len(X_train_df) != len(y_train):
    raise ValueError("Training features and labels have different lengths.")

if len(X_test_df) != len(y_test):
    raise ValueError("Testing features and labels have different lengths.")

if not np.isfinite(X_train_df.to_numpy(dtype=np.float32)).all():
    raise ValueError("Training features contain NaN or infinite values.")

if not np.isfinite(X_test_df.to_numpy(dtype=np.float32)).all():
    raise ValueError("Testing features contain NaN or infinite values.")

if not set(np.unique(y_train)).issubset({0, 1}):
    raise ValueError("Training labels must be binary: 0 or 1.")

if not set(np.unique(y_test)).issubset({0, 1}):
    raise ValueError("Testing labels must be binary: 0 or 1.")

# --------------------------------------------------
# 5. CONVERT DATA TO NUMPY
# --------------------------------------------------

X_train = X_train_df.to_numpy(dtype=np.float32)
X_test = X_test_df.to_numpy(dtype=np.float32)

feature_names = X_train_df.columns.tolist()

# --------------------------------------------------
# 6. TRAIN / VALIDATION SPLIT
# --------------------------------------------------

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=y_train
)

print("\nData split:")
print("Training samples:", len(X_fit))
print("Validation samples:", len(X_val))
print("Final test samples:", len(X_test))

# --------------------------------------------------
# 7. CREATE PYTORCH DATASETS
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(X_fit, dtype=torch.float32),
    torch.tensor(y_fit, dtype=torch.long)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# 8. INITIALIZE TRANSFORMER
# --------------------------------------------------

model = TabularTransformer(
    input_features=INPUT_FEATURES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

print("\nModel initialized successfully.")
print("Total parameters:", sum(p.numel() for p in model.parameters()))

# --------------------------------------------------
# 9. TRAINING AND VALIDATION
# --------------------------------------------------

best_val_loss = float("inf")
best_state = None
best_epoch = 0
epochs_without_improvement = 0

history = []

for epoch in range(1, EPOCHS + 1):

    # Training
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = batch_X.size(0)
        train_loss_sum += loss.item() * batch_size

        predictions = torch.argmax(outputs, dim=1)
        train_correct += (predictions == batch_y).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # Validation: used for model selection only
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            batch_size = batch_X.size(0)
            val_loss_sum += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)
            val_correct += (predictions == batch_y).sum().item()
            val_total += batch_size

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "learning_rate": optimizer.param_groups[0]["lr"]
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

if best_state is None:
    raise RuntimeError("Training failed: no valid model checkpoint was created.")

# Restore the best validation checkpoint
model.load_state_dict(best_state)
model.eval()

print("\nBest epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")

# --------------------------------------------------
# 10. FINAL TEST EVALUATION — ONCE
# --------------------------------------------------

all_predictions = []
all_probabilities = []
all_actual = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(DEVICE)

        outputs = model(batch_X)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_actual.extend(batch_y.numpy())

y_pred = np.asarray(all_predictions, dtype=np.int64)
y_actual = np.asarray(all_actual, dtype=np.int64)
y_probabilities = np.asarray(all_probabilities)

accuracy = accuracy_score(y_actual, y_pred)
precision = precision_score(
    y_actual, y_pred, zero_division=0
)
recall = recall_score(
    y_actual, y_pred, zero_division=0
)
f1 = f1_score(
    y_actual, y_pred, zero_division=0
)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_actual,
        y_pred,
        labels=[0, 1],
        target_names=["Normal Traffic", "Attack Traffic"],
        zero_division=0
    )
)

# --------------------------------------------------
# 11. SAVE MODEL CHECKPOINT
# --------------------------------------------------

checkpoint_path = os.path.join(
    MODEL_DIR, "transformer_model.pt"
)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_features": INPUT_FEATURES,
    "d_model": D_MODEL,
    "nhead": NHEAD,
    "num_layers": NUM_LAYERS,
    "dim_feedforward": DIM_FEEDFORWARD,
    "dropout": DROPOUT,
    "feature_names": feature_names,
    "best_epoch": best_epoch,
    "best_validation_loss": best_val_loss,
    "test_accuracy": accuracy
}, checkpoint_path)

print("\nModel saved to:", checkpoint_path)

# --------------------------------------------------
# 12. SAVE TRAINING HISTORY AND METRICS
# --------------------------------------------------

history_df = pd.DataFrame(history)
history_path = os.path.join(
    REPORT_DIR, "transformer_training_history.csv"
)
history_df.to_csv(history_path, index=False)

metrics_df = pd.DataFrame([{
    "model": "Tabular Transformer",
    "best_epoch": best_epoch,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "best_validation_loss": best_val_loss
}])

metrics_path = os.path.join(
    REPORT_DIR, "transformer_evaluation.csv"
)
metrics_df.to_csv(metrics_path, index=False)

print("Training history saved to:", history_path)
print("Evaluation metrics saved to:", metrics_path)

# --------------------------------------------------
# 13. SAVE CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_actual, y_pred, labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(7, 5))
image = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(image, ax=ax)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
    xlabel="Predicted Label",
    ylabel="Actual Label",
    title="Transformer Confusion Matrix"
)

threshold = cm.max() / 2.0 if cm.size else 0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

fig.tight_layout()

cm_path = os.path.join(
    FIGURE_DIR, "confusion_matrix_transformer.png"
)
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Confusion matrix saved to:", cm_path)

# --------------------------------------------------
# 14. SAVE TRAINING / VALIDATION LOSS GRAPH
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)

ax.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation Loss"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Transformer Training and Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)

loss_path = os.path.join(
    FIGURE_DIR, "transformer_training_loss.png"
)
fig.tight_layout()
fig.savefig(loss_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Loss graph saved to:", loss_path)

# --------------------------------------------------
# 15. FINISHED
# --------------------------------------------------

print("\n" + "=" * 65)
print("TRANSFORMER TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
print("Check saved_models/ and reports/ for output files.")


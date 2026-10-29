import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

processed_dir = os.path.join(
    PROJECT_ROOT,
    "processed_data"
)

selected_data_dir = os.path.join(
    processed_dir,
    "selected_features"
)

models_dir = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

reports_dir = os.path.join(
    PROJECT_ROOT,
    "reports"
)

figures_dir = os.path.join(
    reports_dir,
    "figures"
)

os.makedirs(figures_dir, exist_ok=True)


# --------------------------------------------------
# Header
# --------------------------------------------------

print("=" * 60)
print("FEATURE-SELECTED XGBOOST EVALUATION")
print("=" * 60)


# --------------------------------------------------
# Load Selected Test Data
# --------------------------------------------------

X_test = pd.read_csv(
    os.path.join(
        selected_data_dir,
        "X_test_selected.csv"
    )
)

y_test = pd.read_csv(
    os.path.join(
        processed_dir,
        "y_test.csv"
    )
).squeeze()

print("\nX_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)


# --------------------------------------------------
# Load Feature-Selected Model
# --------------------------------------------------

model_path = os.path.join(
    models_dir,
    "xgboost_feature_selected.pkl"
)

model = joblib.load(model_path)

print("\nFeature-selected XGBoost model loaded successfully.")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# --------------------------------------------------
# Classification Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE-SELECTED XGBOOST RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Attack"
    ]
)

display.plot()

plt.title(
    "Feature-Selected XGBoost Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    figures_dir,
    "confusion_matrix_feature_selected_xgboost.png"
)

plt.savefig(cm_path)

plt.show()

print("\nConfusion Matrix saved at:")
print(cm_path)


# --------------------------------------------------
# Compare Models
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Original XGBoost",
        "Feature-Selected XGBoost"
    ],
    "Features": [
        122,
        50
    ],
    "Accuracy": [
        79.58,
        accuracy * 100
    ],
    "Precision": [
        96.91,
        precision * 100
    ],
    "Recall": [
        66.24,
        recall * 100
    ],
    "F1 Score": [
        78.69,
        f1 * 100
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))


# --------------------------------------------------
# Save Comparison
# --------------------------------------------------

comparison_path = os.path.join(
    reports_dir,
    "final_xgboost_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nComparison saved at:")
print(comparison_path)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE-SELECTED EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)
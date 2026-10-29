import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "processed_data"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

FIGURES_DIR = os.path.join(
    REPORTS_DIR,
    "figures"
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------
# Load Test Data
# --------------------------------------------------

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "y_test.csv")
).squeeze()

print("\nX_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "KNN": "knn.pkl",
    "SVM": "svm.pkl"
}


# --------------------------------------------------
# Evaluate Models
# --------------------------------------------------

results = []

print("\n" + "=" * 60)
print("MODEL EVALUATION STARTED")
print("=" * 60)


for model_name, model_file in models.items():

    print(f"\nEvaluating: {model_name}")

    # Load model
    model_path = os.path.join(
        MODELS_DIR,
        model_file
    )

    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
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

    # Save metrics
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

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

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # Confusion Matrix Figure
    # --------------------------------------------------

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(
        [0, 1],
        ["Normal", "Attack"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Attack"]
    )

    # Display values inside matrix
    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    # File name
    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    figure_path = os.path.join(
        FIGURES_DIR,
        f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    print(
        f"Confusion Matrix saved: {figure_path}"
    )


# --------------------------------------------------
# Create Evaluation DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Convert metrics to percentage
for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:

    results_df[column] = (
        results_df[column] * 100
    )


# --------------------------------------------------
# Save Evaluation Results
# --------------------------------------------------

results_path = os.path.join(
    REPORTS_DIR,
    "evaluation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# Display Final Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved at:")
print(results_path)

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)
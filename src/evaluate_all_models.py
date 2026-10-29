import os
import pandas as pd
import joblib

from sklearn.metrics import (
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

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "processed_data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

# --------------------------------------------------
# Load Test Data
# --------------------------------------------------

print("=" * 60)
print("EVALUATING ALL MACHINE LEARNING MODELS")
print("=" * 60)

X_test = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "X_test.csv"
    )
)
X_test = X_test.values

y_test = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "y_test.csv"
    )
).squeeze()

print("\nX_test:", X_test.shape)
print("y_test:", y_test.shape)


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


results = []


# --------------------------------------------------
# Evaluate Models
# --------------------------------------------------

for model_name, model_file in models.items():

    print("\n" + "-" * 60)
    print("Evaluating:", model_name)

    model_path = os.path.join(
        MODEL_DIR,
        model_file
    )

    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        continue

    model = joblib.load(
        model_path
    )

    y_pred = model.predict(
        X_test
    )

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

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    })


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)
print("FINAL ML METRICS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# Save Results
# --------------------------------------------------

output_path = os.path.join(
    REPORT_DIR,
    "all_ml_metrics.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nResults saved at:")
print(output_path)

print("\n" + "=" * 60)
print("EVALUATION COMPLETED!")
print("=" * 60)
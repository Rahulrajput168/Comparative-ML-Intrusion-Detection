import os
import pandas as pd
import joblib

from xgboost import XGBClassifier
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

processed_dir = os.path.join(
    PROJECT_ROOT,
    "processed_data"
)

models_dir = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

reports_dir = os.path.join(
    PROJECT_ROOT,
    "reports"
)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

print("=" * 60)
print("XGBOOST FEATURE SELECTION")
print("=" * 60)

print("\nLoading processed data...")

X_train = pd.read_csv(
    os.path.join(processed_dir, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(processed_dir, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(processed_dir, "y_train.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(processed_dir, "y_test.csv")
).squeeze()

print("\nOriginal Features:", X_train.shape[1])


# --------------------------------------------------
# Load Existing XGBoost Model
# --------------------------------------------------

model_path = os.path.join(
    models_dir,
    "xgboost_model.pkl"
)

model = joblib.load(model_path)

print("XGBoost model loaded successfully.")


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# --------------------------------------------------
# Select Top Features
# --------------------------------------------------

TOP_FEATURES = 50

selected_features = (
    importance_df
    .head(TOP_FEATURES)["Feature"]
    .tolist()
)

print("\nSelected Features:", len(selected_features))

print("\nTop Selected Features:")
for i, feature in enumerate(
    selected_features[:20],
    start=1
):
    print(f"{i}. {feature}")


# --------------------------------------------------
# Create Reduced Dataset
# --------------------------------------------------

X_train_selected = X_train[
    selected_features
]

X_test_selected = X_test[
    selected_features
]

print("\nReduced Data Shapes:")
print(
    "X_train:",
    X_train_selected.shape
)

print(
    "X_test :",
    X_test_selected.shape
)


# --------------------------------------------------
# Train XGBoost on Selected Features
# --------------------------------------------------

print("\nTraining XGBoost with selected features...")

selected_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

selected_model.fit(
    X_train_selected,
    y_train
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = selected_model.predict(
    X_test_selected
)


# --------------------------------------------------
# Evaluation
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

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# --------------------------------------------------
# Save Selected Features
# --------------------------------------------------

features_path = os.path.join(
    reports_dir,
    "selected_features.csv"
)

pd.DataFrame({
    "Selected Feature": selected_features
}).to_csv(
    features_path,
    index=False
)

print("\nSelected features saved at:")
print(features_path)


# --------------------------------------------------
# Save Reduced Data
# --------------------------------------------------

selected_data_dir = os.path.join(
    PROJECT_ROOT,
    "processed_data",
    "selected_features"
)

os.makedirs(
    selected_data_dir,
    exist_ok=True
)

X_train_selected.to_csv(
    os.path.join(
        selected_data_dir,
        "X_train_selected.csv"
    ),
    index=False
)

X_test_selected.to_csv(
    os.path.join(
        selected_data_dir,
        "X_test_selected.csv"
    ),
    index=False
)

print("\nReduced datasets saved in:")
print(selected_data_dir)


# --------------------------------------------------
# Save Selected Model
# --------------------------------------------------

selected_model_path = os.path.join(
    models_dir,
    "xgboost_feature_selected.pkl"
)

joblib.dump(
    selected_model,
    selected_model_path
)

print("\nFeature-selected model saved at:")
print(selected_model_path)


# --------------------------------------------------
# Save Results
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "XGBoost - All Features",
        "XGBoost - Top 50 Features"
    ],
    "Features": [
        X_train.shape[1],
        len(selected_features)
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

results_path = os.path.join(
    reports_dir,
    "feature_selection_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nComparison results saved at:")
print(results_path)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
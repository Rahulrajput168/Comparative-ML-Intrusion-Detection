import os
import time
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
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
print("XGBOOST HYPERPARAMETER OPTIMIZATION")
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

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)


# --------------------------------------------------
# Base XGBoost Model
# --------------------------------------------------

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Hyperparameter Search Space
# --------------------------------------------------

param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}


# --------------------------------------------------
# Randomized Search
# --------------------------------------------------

print("\nStarting hyperparameter optimization...")
print("This may take some time.")

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    scoring="f1",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()

search.fit(
    X_train,
    y_train
)

optimization_time = time.time() - start_time


# --------------------------------------------------
# Best Parameters
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(f"{search.best_score_:.4f}")

print(
    f"\nOptimization Time: "
    f"{optimization_time:.2f} seconds"
)


# --------------------------------------------------
# Best Model
# --------------------------------------------------

best_model = search.best_estimator_

print("\nGenerating test predictions...")

y_pred = best_model.predict(X_test)


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
print("OPTIMIZED XGBOOST RESULTS")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# --------------------------------------------------
# Save Optimized Model
# --------------------------------------------------

model_path = os.path.join(
    models_dir,
    "xgboost_optimized.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nOptimized model saved at:")
print(model_path)


# --------------------------------------------------
# Save Results
# --------------------------------------------------

results = pd.DataFrame({
    "Model": ["Optimized XGBoost"],
    "Accuracy": [accuracy * 100],
    "Precision": [precision * 100],
    "Recall": [recall * 100],
    "F1 Score": [f1 * 100],
    "Optimization Time (seconds)": [
        optimization_time
    ]
})

results_path = os.path.join(
    reports_dir,
    "optimized_xgboost_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nResults saved at:")
print(results_path)


print("\n" + "=" * 60)
print("HYPERPARAMETER OPTIMIZATION COMPLETED!")
print("=" * 60)
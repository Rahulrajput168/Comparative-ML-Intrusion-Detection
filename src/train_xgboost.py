import os
import sys
import time

import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


def train_xgboost():

    print("=" * 60)
    print("XGBOOST INTRUSION DETECTION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Paths
    # --------------------------------------------------

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
    # 2. Load Processed Data
    # --------------------------------------------------

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

    print("\nData Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # --------------------------------------------------
    # 3. Create XGBoost Model
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CREATING XGBOOST MODEL")
    print("=" * 60)

    model = XGBClassifier(
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

    # --------------------------------------------------
    # 4. Train Model
    # --------------------------------------------------

    print("\nTraining XGBoost...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    print("\nTraining completed!")
    print(
        f"Training Time: {training_time:.2f} seconds"
    )

    # --------------------------------------------------
    # 5. Prediction
    # --------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 6. Evaluation
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

    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # --------------------------------------------------
    # 7. Classification Report
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
    # 8. Save Model
    # --------------------------------------------------

    model_path = os.path.join(
        models_dir,
        "xgboost_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved at:")
    print(model_path)

    # --------------------------------------------------
    # 9. Save Results
    # --------------------------------------------------

    results = pd.DataFrame({
        "Model": ["XGBoost"],
        "Accuracy": [accuracy * 100],
        "Precision": [precision * 100],
        "Recall": [recall * 100],
        "F1 Score": [f1 * 100],
        "Training Time (seconds)": [training_time]
    })

    results_path = os.path.join(
        reports_dir,
        "xgboost_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print("\nResults saved at:")
    print(results_path)

    # --------------------------------------------------
    # 10. Completion Message
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    train_xgboost()
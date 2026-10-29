import os
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

processed_dir = os.path.join(
    PROJECT_ROOT,
    "processed_data"
)

models_dir = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

figures_dir = os.path.join(
    PROJECT_ROOT,
    "reports",
    "figures"
)

os.makedirs(figures_dir, exist_ok=True)


# --------------------------------------------------
# Load Test Data
# --------------------------------------------------

print("=" * 60)
print("XGBOOST EVALUATION")
print("=" * 60)

X_test = pd.read_csv(
    os.path.join(
        processed_dir,
        "X_test.csv"
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
# Load XGBoost Model
# --------------------------------------------------

model_path = os.path.join(
    models_dir,
    "xgboost_model.pkl"
)

model = joblib.load(model_path)

print("\nXGBoost model loaded successfully.")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)


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
    "XGBoost Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    figures_dir,
    "confusion_matrix_xgboost.png"
)

plt.savefig(cm_path)

plt.show()

print("\nConfusion Matrix saved:")
print(cm_path)


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

importance = model.feature_importances_

feature_names = X_test.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .head(20)
)

print("\nTop 20 Important Features:")
print(feature_importance_df)


# --------------------------------------------------
# Feature Importance Graph
# --------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    feature_importance_df["Feature"][::-1],
    feature_importance_df["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 20 XGBoost Feature Importance"
)

plt.tight_layout()

importance_path = os.path.join(
    figures_dir,
    "xgboost_feature_importance.png"
)

plt.savefig(importance_path)

plt.show()

print("\nFeature Importance graph saved:")
print(importance_path)


# --------------------------------------------------
# Save Feature Importance CSV
# --------------------------------------------------

importance_csv = os.path.join(
    PROJECT_ROOT,
    "reports",
    "xgboost_feature_importance.csv"
)

feature_importance_df.to_csv(
    importance_csv,
    index=False
)

print("\nFeature importance data saved:")
print(importance_csv)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 60)
print("XGBOOST EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)
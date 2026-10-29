import os
import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Project paths
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

os.makedirs(MODELS_DIR, exist_ok=True)


# --------------------------------------------------
# Load Processed Data
# --------------------------------------------------

print("=" * 60)
print("LOADING PROCESSED DATA")
print("=" * 60)

X_train = pd.read_csv(
    os.path.join(PROCESSED_DIR, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(PROCESSED_DIR, "y_train.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "y_test.csv")
).squeeze()

print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# --------------------------------------------------
# Define Models
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    )
}


# --------------------------------------------------
# Train Models
# --------------------------------------------------

results = []

print("\n" + "=" * 60)
print("MODEL TRAINING STARTED")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining: {name}")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Save model
    model_filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = os.path.join(
        MODELS_DIR,
        model_filename
    )

    import joblib

    joblib.dump(
        model,
        model_path
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Training Time (seconds)": training_time
    })

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Saved         : {model_path}")


# --------------------------------------------------
# Save Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    PROJECT_ROOT,
    "reports",
    "model_training_results.csv"
)

os.makedirs(
    os.path.dirname(results_path),
    exist_ok=True
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# Final Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nResults saved at:")
print(results_path)

print("\nModels saved in:")
print(MODELS_DIR)
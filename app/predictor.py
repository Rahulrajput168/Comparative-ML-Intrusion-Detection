import os
import joblib


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# ==================================================
# MODEL DIRECTORY
# ==================================================

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)


# ==================================================
# MODEL FILE MAPPING
# ==================================================

MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "Random Forest":
        "random_forest.pkl",

    "KNN":
        "knn.pkl",

    "SVM":
        "svm.pkl",

    "XGBoost":
        "xgboost_model.pkl",

    "Feature-Selected XGBoost":
        "xgboost_feature_selected.pkl",

    "Deep Learning - MLP":
        "deep_learning_mlp.pkl"
}


# ==================================================
# LOAD MODEL
# ==================================================

def load_model(model_name):

    if model_name not in MODEL_FILES:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_file = MODEL_FILES[
        model_name
    ]

    model_path = os.path.join(
        MODEL_DIR,
        model_file
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    return model


# ==================================================
# PREDICTION LABEL
# ==================================================

def prediction_label(prediction):

    if int(prediction) == 0:

        return "Normal Traffic"

    else:

        return "Attack Traffic"
import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# PROJECT ROOT AND IMPORTS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformer_model import TabularTransformer

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Comparative Analysis of ML Algorithms",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# DIRECTORIES
# ============================================================

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
SELECTED_FEATURE_DIR = os.path.join(PROCESSED_DIR, "selected_features")

TRANSFORMER_PATH = os.path.join(
    SAVED_MODELS_DIR, "transformer_model.pt"
)

# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": os.path.join(SAVED_MODELS_DIR, "logistic_regression.pkl"),
    "Decision Tree": os.path.join(SAVED_MODELS_DIR, "decision_tree.pkl"),
    "Random Forest": os.path.join(SAVED_MODELS_DIR, "random_forest.pkl"),
    "KNN": os.path.join(SAVED_MODELS_DIR, "knn.pkl"),
    "SVM": os.path.join(SAVED_MODELS_DIR, "svm.pkl"),
    "XGBoost": os.path.join(SAVED_MODELS_DIR, "xgboost_model.pkl"),
    "Feature-Selected XGBoost": os.path.join(
        SAVED_MODELS_DIR, "xgboost_feature_selected.pkl"
    ),
    "Deep Learning - MLP": os.path.join(
        SAVED_MODELS_DIR, "deep_learning_mlp.pkl"
    ),
    "Optimized XGBoost": os.path.join(
        SAVED_MODELS_DIR, "xgboost_optimized.pkl"
    ),
}

MODEL_FEATURE_COUNT = {
    "Logistic Regression": 122,
    "Decision Tree": 122,
    "Random Forest": 122,
    "KNN": 122,
    "SVM": 122,
    "XGBoost": 122,
    "Feature-Selected XGBoost": 50,
    "Deep Learning - MLP": 122,
    "Optimized XGBoost": 122,
    "Tabular Transformer": 122,
}

# ============================================================
# LOAD MODEL COMPARISON RESULTS
# ============================================================

results_path = os.path.join(REPORTS_DIR, "final_model_comparison.csv")

if not os.path.exists(results_path):
    st.error(f"Model comparison file not found:\n{results_path}")
    st.stop()

results_df = pd.read_csv(results_path)

required_columns = [
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
]

missing_columns = [
    col for col in required_columns if col not in results_df.columns
]

if missing_columns:
    st.error(f"Missing columns in comparison CSV: {missing_columns}")
    st.stop()

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_test_data():
    X_path = os.path.join(PROCESSED_DIR, "X_test.csv")
    y_path = os.path.join(PROCESSED_DIR, "y_test.csv")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("X_test.csv or y_test.csv was not found.")

    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path).squeeze("columns")

    return X_test, y_test


@st.cache_data
def load_selected_test_data():
    path = os.path.join(SELECTED_FEATURE_DIR, "X_test_selected.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected test dataset not found:\n{path}")

    return pd.read_csv(path)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_prediction_model(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    path = MODEL_PATHS[model_name]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found:\n{path}")

    return joblib.load(path)


@st.cache_resource
def load_transformer_model():
    if not os.path.exists(TRANSFORMER_PATH):
        raise FileNotFoundError(
            f"Transformer checkpoint not found:\n{TRANSFORMER_PATH}"
        )

    try:
        checkpoint = torch.load(
            TRANSFORMER_PATH, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(TRANSFORMER_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        input_features = checkpoint.get("input_features", 122)
        d_model = checkpoint.get("d_model", 64)
        nhead = checkpoint.get("nhead", 4)
        num_layers = checkpoint.get("num_layers", 2)
        dim_feedforward = checkpoint.get("dim_feedforward", 128)
        dropout = checkpoint.get("dropout", 0.2)
    else:
        state_dict = checkpoint
        input_features = 122
        d_model = 64
        nhead = 4
        num_layers = 2
        dim_feedforward = 128
        dropout = 0.2

    model = TabularTransformer(
        input_features=input_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# PREDICTION HELPERS
# ============================================================

def prediction_label(prediction):
    return "Normal Traffic" if int(prediction) == 0 else "Attack Traffic"



def prepare_sample_for_model(model, sample):
    """
    Prepare test data in the feature order expected by the model.
    Supports both named features and models trained with numbered columns.
    """
    sample = sample.copy()

    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

        # Case 1: The model expects the same named columns.
        if all(col in sample.columns for col in expected_columns):
            return sample[expected_columns]

        # Case 2: The model was trained with numbered columns
        # such as 0, 1, 2, ... or "0", "1", "2", ...
        numbered_columns = [
            str(col) for col in expected_columns
        ]

        is_numbered_features = numbered_columns == [
            str(i) for i in range(len(expected_columns))
        ]

        if (
            is_numbered_features
            and sample.shape[1] == len(expected_columns)
        ):
            # Preserve the original test-data column order.
            # Passing NumPy data avoids a column-name mismatch.
            return sample.to_numpy()

        missing = [
            col for col in expected_columns
            if col not in sample.columns
        ]

        raise ValueError(
            "Feature mismatch. The model expects named columns "
            f"that are not present in the dataset: {missing[:20]}. "
            f"Dataset columns: {list(sample.columns[:20])}"
        )

    # Models without saved feature names.
    expected_count = getattr(model, "n_features_in_", None)

    if expected_count is not None and sample.shape[1] != expected_count:
        raise ValueError(
            f"Model expects {expected_count} features, "
            f"but received {sample.shape[1]}."
        )

    return sample.to_numpy()



def transformer_predict(model, data):
    if data.shape[1] != 122:
        raise ValueError(
            f"Tabular Transformer requires 122 features; received {data.shape[1]}."
        )

    values = data.to_numpy(dtype=np.float32)
    tensor = torch.tensor(values, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return predictions.cpu().numpy(), probabilities.cpu().numpy()


def predict_with_model(model_name, data):
    if model_name == "Tabular Transformer":
        model = load_transformer_model()
        return transformer_predict(model, data)

    model = load_prediction_model(model_name)
    prepared = prepare_sample_for_model(model, data)
    predictions = model.predict(prepared)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prepared)
    else:
        probabilities = None

    return np.asarray(predictions), probabilities


def get_confusion_matrix_path(model_name):
    filenames = {
        "Logistic Regression": "confusion_matrix_logistic_regression.png",
        "Decision Tree": "confusion_matrix_decision_tree.png",
        "Random Forest": "confusion_matrix_random_forest.png",
        "KNN": "confusion_matrix_knn.png",
        "SVM": "confusion_matrix_svm.png",
        "XGBoost": "confusion_matrix_xgboost.png",
        "Feature-Selected XGBoost": "confusion_matrix_feature_selected_xgboost.png",
        "Deep Learning - MLP": "confusion_matrix_deep_learning.png",
        "Tabular Transformer": "confusion_matrix_transformer.png",
        "Optimized XGBoost": "confusion_matrix_optimized_xgboost.png",
    }

    filename = filenames.get(model_name)

    if filename is None:
        return None

    return os.path.join(FIGURES_DIR, filename)


def generate_confusion_matrix(model_name):
    """Generate a real confusion matrix from the saved model and test data."""
    X_test, y_test = load_test_data()

    if model_name == "Feature-Selected XGBoost":
        X_test = load_selected_test_data()

    predictions, _ = predict_with_model(model_name, X_test)

    if len(predictions) != len(y_test):
        raise ValueError(
            "Prediction count does not match the test-label count."
        )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    output_path = get_confusion_matrix_path(model_name)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Intrusion Detection")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📊 Model Comparison",
        "🔍 Confusion Matrix",
        "🚨 Single Prediction",
        "📁 CSV Prediction",
        "📈 XGBoost Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("### Project Details")
st.sidebar.write("📌 Dataset: NSL-KDD")
st.sidebar.write("📌 Original Features: 122")
st.sidebar.write("📌 Selected Features: 50")
st.sidebar.write(f"📌 Models: {len(results_df)}")

best_model = results_df.loc[results_df["F1 Score"].idxmax()]
st.sidebar.write(f"📌 Best Model: {best_model['Model']}")

st.sidebar.markdown("---")
st.sidebar.success("Developed by Rahul Singh")
st.sidebar.caption("B.Tech CSE Project — Intrusion Detection System")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.title("🛡️ Comparative Analysis of Machine Learning Algorithms for Intrusion Detection")

    st.markdown("""
    ### Intrusion Detection System

    This dashboard compares traditional Machine Learning, XGBoost,
    feature-selected XGBoost and deep learning models for detecting
    **Normal** and **Attack** network traffic.
    """)

    st.divider()
    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", "125,973")
    c2.metric("Testing Samples", "22,544")
    c3.metric("Original Features", "122")
    c4.metric("Models Evaluated", len(results_df))

    st.divider()
    st.header("🏆 Best Performing Model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", str(best_model["Model"]))
    c2.metric("Accuracy", f"{best_model['Accuracy']:.2f}%")
    c3.metric("Recall", f"{best_model['Recall']:.2f}%")
    c4.metric("F1 Score", f"{best_model['F1 Score']:.2f}%")

    display_df = results_df.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    display_df[metric_columns] = display_df[metric_columns].round(2)

    st.divider()
    st.subheader("Detailed Model Results")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    chart_path = os.path.join(
        FIGURES_DIR, "final_model_performance_comparison.png"
    )

    if os.path.exists(chart_path):
        st.subheader("📈 Model Performance Comparison")
        st.image(chart_path, use_container_width=True)
    else:
        st.info("Final model performance graph was not found.")


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.write("Compare the performance metrics of all evaluated models.")
    st.divider()

    metric = st.selectbox(
        "Select Performance Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    )

    st.bar_chart(results_df.set_index("Model")[[metric]])

    accuracy_chart = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
    if os.path.exists(accuracy_chart):
        st.subheader("Accuracy Comparison")
        st.image(accuracy_chart, use_container_width=True)

    st.divider()
    st.subheader("Detailed Model Results")
    st.dataframe(
        results_df.round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

elif page == "🔍 Confusion Matrix":
    st.title("🔍 Confusion Matrix")
    st.write("View a model's confusion matrix and evaluation metrics.")
    st.divider()

    available_models = results_df["Model"].astype(str).tolist()
    model_name = st.selectbox("Select Model", available_models)

    cm_path = get_confusion_matrix_path(model_name)

    if cm_path and os.path.exists(cm_path):
        st.image(
            cm_path,
            caption=f"Confusion Matrix — {model_name}",
            width=700
        )
    elif model_name == "Optimized XGBoost":
        st.info(
            "The Optimized XGBoost confusion matrix does not exist yet. "
            "Generate it from the model's test predictions below."
        )

        if st.button("Generate Optimized XGBoost Confusion Matrix", type="primary"):
            try:
                with st.spinner("Generating confusion matrix from test data..."):
                    cm_path = generate_confusion_matrix(model_name)
                st.success("Confusion matrix generated successfully.")
                st.image(cm_path, caption=f"Confusion Matrix — {model_name}", width=700)
            except Exception as e:
                st.error(f"Could not generate confusion matrix:\n{e}")
    else:
        st.warning("Confusion matrix image was not found.")

        if st.button(f"Generate matrix for {model_name}"):
            try:
                with st.spinner("Generating confusion matrix..."):
                    cm_path = generate_confusion_matrix(model_name)
                st.success("Confusion matrix generated successfully.")
                st.image(cm_path, caption=f"Confusion Matrix — {model_name}", width=700)
            except Exception as e:
                st.error(f"Could not generate confusion matrix:\n{e}")

    selected_rows = results_df[results_df["Model"] == model_name]

    if not selected_rows.empty:
        selected_result = selected_rows.iloc[0]
        st.divider()
        st.subheader(f"📊 {model_name} Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{selected_result['Accuracy']:.2f}%")
        c2.metric("Precision", f"{selected_result['Precision']:.2f}%")
        c3.metric("Recall", f"{selected_result['Recall']:.2f}%")
        c4.metric("F1 Score", f"{selected_result['F1 Score']:.2f}%")


# ============================================================
# SINGLE PREDICTION
# ============================================================

elif page == "🚨 Single Prediction":
    st.title("🚨 Intrusion Prediction")
    st.write("Select a trained model and a sample from the test dataset.")
    st.divider()

    try:
        X_test, y_test = load_test_data()
    except Exception as e:
        st.error(f"Unable to load test data:\n{e}")
        st.stop()

    prediction_model = st.selectbox(
        "🤖 Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "SVM",
            "XGBoost",
            "Feature-Selected XGBoost",
            "Deep Learning - MLP",
            "Optimized XGBoost",
            "Tabular Transformer"
        ]
    )

    sample_number = st.number_input(
        "🔢 Select Test Sample Number",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )
    sample_number = int(sample_number)

    try:
        if prediction_model == "Feature-Selected XGBoost":
            X_selected = load_selected_test_data()
            current_sample = X_selected.iloc[[sample_number]]
            expected_features = 50
        else:
            current_sample = X_test.iloc[[sample_number]]
            expected_features = 122

        st.info(
            f"Selected Model: **{prediction_model}**  \n"
            f"Expected Features: **{expected_features}**  \n"
            f"Sample Features: **{current_sample.shape[1]}**"
        )

        with st.expander("👁️ View Selected Test Sample"):
            st.dataframe(current_sample, use_container_width=True)

        if st.button("🔍 Predict Intrusion", type="primary", use_container_width=True):
            with st.spinner(f"Running {prediction_model}..."):
                predictions, probabilities = predict_with_model(
                    prediction_model, current_sample
                )

            prediction = int(predictions[0])
            actual_value = int(y_test.iloc[sample_number])
            predicted_label = prediction_label(prediction)
            actual_label = prediction_label(actual_value)

            st.subheader("🎯 Prediction Result")
            if prediction == 0:
                st.success(f"🟢 Prediction: **{predicted_label}**")
            else:
                st.error(f"🔴 Prediction: **{predicted_label}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Class", predicted_label)
            c2.metric("Actual Class", actual_label)
            c3.metric(
                "Result",
                "Correct ✅" if prediction == actual_value else "Incorrect ⚠️"
            )

            if prediction == actual_value:
                st.success("Correct prediction.")
            else:
                st.warning("The prediction does not match the actual test label.")

            if probabilities is not None:
                confidence = float(np.max(probabilities[0]) * 100)
                st.metric("Prediction Confidence", f"{confidence:.2f}%")

            st.caption(
                "Confidence is the model's predicted probability, not a guarantee."
            )

    except Exception as e:
        st.error(f"Prediction error:\n{e}")


# ============================================================
# CSV PREDICTION
# ============================================================

elif page == "📁 CSV Prediction":
    st.title("📁 CSV File Upload Prediction")
    st.write("Upload a processed CSV containing the required model features.")
    st.divider()

    csv_model = st.selectbox(
        "🤖 Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN",
            "SVM",
            "XGBoost",
            "Feature-Selected XGBoost",
            "Deep Learning - MLP",
            "Optimized XGBoost",
            "Tabular Transformer"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload Processed CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.write(
                f"Rows: **{uploaded_df.shape[0]}** | "
                f"Columns: **{uploaded_df.shape[1]}**"
            )
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            required_features = MODEL_FEATURE_COUNT[csv_model]

            if uploaded_df.shape[1] != required_features:
                st.error(
                    f"{csv_model} requires {required_features} feature columns, "
                    f"but the uploaded file contains {uploaded_df.shape[1]}."
                )
            elif st.button("🚨 Predict Uploaded Data", type="primary"):
                with st.spinner("Generating predictions..."):
                    predictions, probabilities = predict_with_model(
                        csv_model, uploaded_df
                    )

                result_df = uploaded_df.copy()
                result_df["Prediction"] = [
                    prediction_label(p) for p in predictions
                ]

                if probabilities is not None:
                    result_df["Confidence (%)"] = (
                        np.max(probabilities, axis=1) * 100
                    ).round(2)

                normal_count = sum(p == 0 for p in predictions)
                attack_count = sum(p == 1 for p in predictions)

                st.subheader("📊 Prediction Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records", len(predictions))
                c2.metric("Normal Traffic", normal_count)
                c3.metric("Attack Traffic", attack_count)

                st.subheader("🔍 Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                st.download_button(
                    "⬇️ Download Prediction Results",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="intrusion_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"CSV prediction error:\n{e}")


# ============================================================
# XGBOOST ANALYSIS
# ============================================================

elif page == "📈 XGBoost Analysis":
    st.title("📈 XGBoost & Feature Selection Analysis")
    st.write("Compare XGBoost with its feature-selected version.")
    st.divider()

    for model_name in ["XGBoost", "Feature-Selected XGBoost"]:
        row = results_df[results_df["Model"] == model_name]

        if not row.empty:
            result = row.iloc[0]
            st.subheader(model_name)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{result['Accuracy']:.2f}%")
            c2.metric("Precision", f"{result['Precision']:.2f}%")
            c3.metric("Recall", f"{result['Recall']:.2f}%")
            c4.metric("F1 Score", f"{result['F1 Score']:.2f}%")
            st.divider()

    st.subheader("🔻 Feature Reduction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Features", "122")
    c2.metric("Selected Features", "50")
    c3.metric("Feature Reduction", "59.02%")

    reduction_chart = os.path.join(
        FIGURES_DIR, "xgboost_feature_reduction.png"
    )
    if os.path.exists(reduction_chart):
        st.image(reduction_chart, use_container_width=True)

    importance_chart = os.path.join(
        FIGURES_DIR, "xgboost_feature_importance.png"
    )
    if os.path.exists(importance_chart):
        st.subheader("⭐ XGBoost Feature Importance")
        st.image(importance_chart, use_container_width=True)
    else:
        st.info("XGBoost feature importance graph was not found.")

    selected_features_path = os.path.join(REPORTS_DIR, "selected_features.csv")
    if os.path.exists(selected_features_path):
        st.subheader("📋 Selected Features")
        selected_features = pd.read_csv(selected_features_path)
        st.dataframe(
            selected_features.head(50),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.caption("ML Intrusion Detection System")
st.sidebar.caption("NSL-KDD Dataset • B.Tech CSE Project")

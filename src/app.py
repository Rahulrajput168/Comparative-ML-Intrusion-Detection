import os
import sys
import joblib
import streamlit as st
import pandas as pd
import torch
import numpy as np

# ============================================================
# PROJECT PATHS AND IMPORTS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformer_model import TabularTransformer

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
SELECTED_FEATURE_DIR = os.path.join(
    PROCESSED_DIR, "selected_features"
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Comparative Analysis of ML Algorithms",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD FINAL COMPARISON RESULTS
# Use the final CSV as the single source of comparison metrics.
# The Transformer metrics are already included in this CSV.
# ============================================================

RESULTS_PATH = os.path.join(
    REPORTS_DIR, "final_model_comparison.csv"
)

if not os.path.exists(RESULTS_PATH):
    st.error(f"Comparison file not found:\n{RESULTS_PATH}")
    st.stop()

try:
    results_df = pd.read_csv(RESULTS_PATH)
except Exception as exc:
    st.error(f"Could not read comparison results: {exc}")
    st.stop()

REQUIRED_COLUMNS = [
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
]

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in results_df.columns
]

if missing_columns:
    st.error(f"Missing columns in comparison CSV: {missing_columns}")
    st.stop()

for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    results_df[column] = pd.to_numeric(
        results_df[column], errors="coerce"
    )

results_df = results_df.dropna(
    subset=REQUIRED_COLUMNS
).copy()

results_df["Model"] = results_df["Model"].astype(str).str.strip()
results_df = results_df.drop_duplicates(
    subset=["Model"], keep="last"
).reset_index(drop=True)

if results_df.empty:
    st.error("The comparison CSV does not contain valid model results.")
    st.stop()

# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": os.path.join(
        SAVED_MODELS_DIR, "logistic_regression.pkl"
    ),
    "Decision Tree": os.path.join(
        SAVED_MODELS_DIR, "decision_tree.pkl"
    ),
    "Random Forest": os.path.join(
        SAVED_MODELS_DIR, "random_forest.pkl"
    ),
    "KNN": os.path.join(
        SAVED_MODELS_DIR, "knn.pkl"
    ),
    "SVM": os.path.join(
        SAVED_MODELS_DIR, "svm.pkl"
    ),
    "XGBoost": os.path.join(
        SAVED_MODELS_DIR, "xgboost_model.pkl"
    ),
    "Feature-Selected XGBoost": os.path.join(
        SAVED_MODELS_DIR, "xgboost_feature_selected.pkl"
    ),
    "Deep Learning - MLP": os.path.join(
        SAVED_MODELS_DIR, "deep_learning_mlp.pkl"
    ),
    "Optimized XGBoost": os.path.join(
        SAVED_MODELS_DIR, "xgboost_optimized.pkl"
    )
}

TRANSFORMER_MODEL_PATH = os.path.join(
    SAVED_MODELS_DIR, "transformer_model.pt"
)

TRANSFORMER_NAME = "Tabular Transformer"

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
    TRANSFORMER_NAME: 122
}

# Only expose prediction options with a configured loading path.
PREDICTION_MODELS = [
    name for name in MODEL_FEATURE_COUNT
    if name == TRANSFORMER_NAME
    or name in MODEL_PATHS
]

# ============================================================
# DATA HELPERS
# ============================================================

def load_test_data():
    x_path = os.path.join(PROCESSED_DIR, "X_test.csv")
    y_path = os.path.join(PROCESSED_DIR, "y_test.csv")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Test features not found:\n{x_path}")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Test labels not found:\n{y_path}")

    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()

    return X_test, y_test


def load_selected_test_data():
    path = os.path.join(
        SELECTED_FEATURE_DIR, "X_test_selected.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature-selected test data not found:\n{path}"
        )

    return pd.read_csv(path)


def load_prediction_model(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"No standard model loader is configured for {model_name}."
        )

    path = MODEL_PATHS[model_name]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found:\n{path}")

    return joblib.load(path)


@st.cache_resource
def load_transformer_model():
    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        raise FileNotFoundError(
            f"Transformer checkpoint not found:\n"
            f"{TRANSFORMER_MODEL_PATH}"
        )

    # weights_only=False supports checkpoints saved as dictionaries
    # containing model configuration and state_dict.
    try:
        checkpoint = torch.load(
            TRANSFORMER_MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            TRANSFORMER_MODEL_PATH,
            map_location="cpu"
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = TabularTransformer(
            input_features=checkpoint.get("input_features", 122),
            d_model=checkpoint.get("d_model", 64),
            nhead=checkpoint.get("nhead", 4),
            num_layers=checkpoint.get("num_layers", 2),
            dim_feedforward=checkpoint.get("dim_feedforward", 128),
            dropout=checkpoint.get("dropout", 0.2)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = TabularTransformer(
            input_features=122,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.2
        )
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def transformer_predict(model, sample):
    if isinstance(sample, pd.Series):
        sample = sample.to_frame().T

    if not isinstance(sample, pd.DataFrame):
        sample = pd.DataFrame(sample)

    if sample.shape[1] != 122:
        raise ValueError(
            f"Transformer requires 122 features; "
            f"received {sample.shape[1]}."
        )

    values = sample.to_numpy(dtype=np.float32)
    tensor = torch.tensor(values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

    return (
        predictions.cpu().numpy(),
        probabilities.cpu().numpy()
    )


def prediction_label(prediction):
    return (
        "Normal Traffic"
        if int(prediction) == 0
        else "Attack Traffic"
    )


def prepare_sample_for_model(model, sample):
    if isinstance(sample, pd.Series):
        sample = sample.to_frame().T

    if not isinstance(sample, pd.DataFrame):
        sample = pd.DataFrame(sample)

    expected_count = getattr(model, "n_features_in_", None)

    if (
        expected_count is not None
        and sample.shape[1] != int(expected_count)
    ):
        raise ValueError(
            f"Model expects {int(expected_count)} features, "
            f"but received {sample.shape[1]}."
        )

    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

        if all(column in sample.columns for column in expected_columns):
            return sample.loc[:, expected_columns]

        expected_as_strings = [str(c) for c in expected_columns]
        names_are_numeric_positions = (
            expected_as_strings
            == [str(i) for i in range(len(expected_columns))]
        ) or (
            expected_columns == list(range(len(expected_columns)))
        )

        if (
            names_are_numeric_positions
            and sample.shape[1] == len(expected_columns)
        ):
            return sample.to_numpy()

        missing = [
            str(column) for column in expected_columns
            if column not in sample.columns
        ]

        raise ValueError(
            "Input columns do not match the model's training features. "
            "Use the correctly processed dataset and feature order. "
            f"Missing features: {missing[:20]}"
        )

    return sample.to_numpy()


def get_prediction_confidence(model, sample):
    try:
        prepared = prepare_sample_for_model(model, sample)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(prepared)[0]
            return float(np.max(probabilities) * 100)
    except Exception:
        return None

    return None


def show_metric_cards(row):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{row['Accuracy']:.2f}%")
    with col2:
        st.metric("Precision", f"{row['Precision']:.2f}%")
    with col3:
        st.metric("Recall", f"{row['Recall']:.2f}%")
    with col4:
        st.metric("F1 Score", f"{row['F1 Score']:.2f}%")


def show_figure(filename, caption=None):
    path = os.path.join(FIGURES_DIR, filename)

    if os.path.exists(path):
        st.image(
            path,
            caption=caption,
            use_container_width=True
        )
    else:
        st.info(f"Graph not available yet: {filename}")


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
st.sidebar.write(f"📌 Model Entries: {len(results_df)}")

best_f1_model = results_df.loc[results_df["F1 Score"].idxmax()]
st.sidebar.write(f"📌 Highest F1: {best_f1_model['Model']}")

st.sidebar.markdown("---")
st.sidebar.success("Developed by Rahul Singh")
st.sidebar.caption(
    "Comparative Analysis of Machine Learning Algorithms "
    "for Intrusion Detection"
)
st.sidebar.caption("B.Tech CSE Project")

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.title(
        "🛡️ Comparative Analysis of Machine Learning Algorithms "
        "for Intrusion Detection"
    )

    st.markdown(
        """
        ### Intrusion Detection System

        This dashboard compares traditional Machine Learning,
        XGBoost variants, a Multi-Layer Perceptron (MLP), and a
        Tabular Transformer for classifying network traffic as
        **Normal Traffic** or **Attack Traffic**.
        """
    )

    st.divider()
    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Training Samples", "125,973")
    c2.metric("Testing Samples", "22,544")
    c3.metric("Original Features", "122")
    c4.metric("Model Entries", len(results_df))

    st.divider()
    st.header("🏆 Best Performing Model (by F1 Score)")

    show_metric_cards(best_f1_model)

    st.success(
        f"{best_f1_model['Model']} has the highest F1-score in the "
        f"current comparison CSV: {best_f1_model['F1 Score']:.2f}%."
    )

    st.divider()
    st.header("🤖 Final Model Performance")

    display_df = results_df.copy()
    numeric_columns = [
        "Accuracy", "Precision", "Recall", "F1 Score"
    ]
    display_df[numeric_columns] = display_df[numeric_columns].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.header("📈 Model Performance Comparison")
    show_figure(
        "final_model_performance_comparison.png",
        "Accuracy, precision, recall and F1-score comparison"
    )

# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.write(
        "Explore all model entries and compare their reported metrics."
    )

    metric = st.selectbox(
        "Select Performance Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    )

    chart_data = results_df[["Model", metric]].set_index("Model")
    st.bar_chart(chart_data)

    st.divider()
    st.subheader("All Metrics Comparison")
    show_figure(
        "all_metrics_comparison.png",
        "Grouped comparison of accuracy, precision, recall and F1-score"
    )

    st.divider()
    st.subheader("Accuracy Comparison")
    show_figure("accuracy_comparison.png")

    st.divider()
    st.subheader("Detailed Model Results")
    display_df = results_df.copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================

elif page == "🔍 Confusion Matrix":
    st.title("🔍 Confusion Matrix")
    st.write(
        "View an available confusion matrix and the corresponding "
        "metrics stored in the comparison CSV."
    )

    model_name = st.selectbox(
        "Select Model",
        results_df["Model"].tolist()
    )

    confusion_matrix_files = {
        "Logistic Regression": "confusion_matrix_logistic_regression.png",
        "Decision Tree": "confusion_matrix_decision_tree.png",
        "Random Forest": "confusion_matrix_random_forest.png",
        "KNN": "confusion_matrix_knn.png",
        "SVM": "confusion_matrix_svm.png",
        "XGBoost": "confusion_matrix_xgboost.png",
        "Feature-Selected XGBoost":
            "confusion_matrix_feature_selected_xgboost.png",
        "Deep Learning - MLP": "confusion_matrix_deep_learning.png",
        "Tabular Transformer": "confusion_matrix_transformer.png",
        "Optimized XGBoost": "confusion_matrix_optimized_xgboost.png"
    }

    filename = confusion_matrix_files.get(model_name)

    if filename:
        show_figure(filename, f"Confusion Matrix - {model_name}")
    else:
        st.info("No confusion matrix filename is configured for this model.")

    selected_result = results_df[
        results_df["Model"] == model_name
    ].iloc[0]

    st.divider()
    st.subheader(f"📊 {model_name} Metrics")
    show_metric_cards(selected_result)

# ============================================================
# SINGLE PREDICTION
# ============================================================

elif page == "🚨 Single Prediction":
    st.title("🚨 Intrusion Prediction")
    st.write(
        "Select a configured model and a test sample to compare "
        "the predicted class with the dataset's actual label."
    )

    try:
        X_test, y_test = load_test_data()
    except Exception as exc:
        st.error(f"Unable to load test data:\n{exc}")
        st.stop()

    try:
        X_test_selected = load_selected_test_data()
        selected_data_available = True
    except Exception as exc:
        X_test_selected = None
        selected_data_available = False
        st.warning(f"Feature-selected test data unavailable: {exc}")

    available_prediction_models = [
        name for name in PREDICTION_MODELS
        if name in results_df["Model"].tolist()
    ]

    prediction_model = st.selectbox(
        "🤖 Select Model",
        available_prediction_models
    )

    sample_number = int(
        st.number_input(
            "🔢 Test Sample Number",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1
        )
    )

    if prediction_model == "Feature-Selected XGBoost":
        if not selected_data_available:
            st.error("Feature-selected test data is unavailable.")
            st.stop()

        if sample_number >= len(X_test_selected):
            st.error("Sample number is outside the selected-feature dataset.")
            st.stop()

        current_sample = X_test_selected.iloc[[sample_number]]
    else:
        current_sample = X_test.iloc[[sample_number]]

    expected_features = MODEL_FEATURE_COUNT[prediction_model]

    st.info(
        f"Selected model: **{prediction_model}**  \n"
        f"Required features: **{expected_features}**  \n"
        f"Sample features: **{current_sample.shape[1]}**"
    )

    with st.expander("👁️ View Selected Test Sample"):
        st.dataframe(
            current_sample,
            use_container_width=True,
            hide_index=True
        )

    if st.button(
        "🔍 Predict Intrusion",
        type="primary",
        use_container_width=True
    ):
        try:
            with st.spinner(f"Loading {prediction_model}..."):
                if prediction_model == TRANSFORMER_NAME:
                    model = load_transformer_model()
                else:
                    model = load_prediction_model(prediction_model)

            if current_sample.shape[1] != expected_features:
                raise ValueError(
                    f"{prediction_model} requires {expected_features} "
                    f"features, but received {current_sample.shape[1]}."
                )

            with st.spinner("Generating prediction..."):
                if prediction_model == TRANSFORMER_NAME:
                    predictions, probabilities = transformer_predict(
                        model, current_sample
                    )
                    prediction = int(predictions[0])
                    confidence = float(probabilities[0].max() * 100)
                else:
                    prepared_sample = prepare_sample_for_model(
                        model, current_sample
                    )
                    prediction = int(model.predict(prepared_sample)[0])
                    confidence = get_prediction_confidence(
                        model, current_sample
                    )

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
                "Correct ✅" if prediction == actual_value
                else "Incorrect ⚠️"
            )

            if prediction == actual_value:
                st.success("Correct prediction for this labelled test sample.")
            else:
                st.warning("Prediction differs from the test sample's actual label.")

            if confidence is not None:
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )
                st.caption(
                    "Confidence is the model's highest class probability; "
                    "it is not a guarantee that the prediction is correct."
                )

            model_information = {
                "Logistic Regression": "Logistic Regression classifier.",
                "Decision Tree": "Decision Tree classifier.",
                "Random Forest": "Random Forest classifier.",
                "KNN": "K-Nearest Neighbors classifier.",
                "SVM": "Support Vector Machine classifier.",
                "XGBoost": "XGBoost gradient-boosting classifier.",
                "Feature-Selected XGBoost":
                    "XGBoost model using 50 selected features.",
                "Deep Learning - MLP":
                    "Multi-Layer Perceptron neural network.",
                "Optimized XGBoost":
                    "Optimized XGBoost model loaded from its saved pickle file.",
                "Tabular Transformer":
                    "PyTorch Tabular Transformer model using 122 processed features."
            }

            st.info(model_information.get(
                prediction_model, "Prediction generated by the selected model."
            ))

        except Exception as exc:
            st.error(f"Prediction error:\n{exc}")

# ============================================================
# CSV PREDICTION
# ============================================================

elif page == "📁 CSV Prediction":
    st.title("📁 CSV File Upload Prediction")
    st.write(
        "Upload a processed CSV containing the model's expected features. "
        "The file must use the correct feature order and preprocessing."
    )

    csv_model = st.selectbox(
        "🤖 Select Model",
        [
            name for name in PREDICTION_MODELS
            if name in results_df["Model"].tolist()
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload Processed CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Dataset")
            st.write(
                f"Rows: **{uploaded_df.shape[0]}** | "
                f"Columns: **{uploaded_df.shape[1]}**"
            )

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )

            required_features = MODEL_FEATURE_COUNT[csv_model]

            if uploaded_df.shape[1] != required_features:
                st.error(
                    f"{csv_model} requires {required_features} features, "
                    f"but the uploaded file contains {uploaded_df.shape[1]} columns."
                )
            elif uploaded_df.empty:
                st.error("The uploaded CSV contains no rows.")
            elif st.button(
                "🚨 Predict Uploaded Data",
                type="primary"
            ):
                try:
                    if csv_model == TRANSFORMER_NAME:
                        model = load_transformer_model()
                        predictions, _ = transformer_predict(
                            model, uploaded_df
                        )
                    else:
                        model = load_prediction_model(csv_model)
                        prepared_data = prepare_sample_for_model(
                            model, uploaded_df
                        )
                        predictions = model.predict(prepared_data)

                    prediction_labels = [
                        prediction_label(value)
                        for value in predictions
                    ]

                    result_df = uploaded_df.copy()
                    result_df["Prediction"] = prediction_labels

                    normal_count = prediction_labels.count("Normal Traffic")
                    attack_count = prediction_labels.count("Attack Traffic")

                    st.subheader("📊 Prediction Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", len(prediction_labels))
                    c2.metric("Normal Traffic", normal_count)
                    c3.metric("Attack Traffic", attack_count)

                    st.subheader("🔍 Prediction Results")
                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="intrusion_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as exc:
                    st.error(f"Prediction error:\n{exc}")

        except Exception as exc:
            st.error(f"Could not read uploaded CSV:\n{exc}")

# ============================================================
# XGBOOST ANALYSIS
# ============================================================

elif page == "📈 XGBoost Analysis":
    st.title("📈 XGBoost & Feature Selection Analysis")
    st.write(
        "Compare XGBoost with its feature-selected variant."
    )

    xgb_result = results_df[
        results_df["Model"] == "XGBoost"
    ]
    selected_result = results_df[
        results_df["Model"] == "Feature-Selected XGBoost"
    ]

    c1, c2 = st.columns(2)

    if not xgb_result.empty:
        with c1:
            st.subheader("XGBoost")
            show_metric_cards(xgb_result.iloc[0])

    if not selected_result.empty:
        with c2:
            st.subheader("Feature-Selected XGBoost")
            show_metric_cards(selected_result.iloc[0])

    st.divider()
    st.subheader("🔻 Feature Reduction")

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Features", 122)
    c2.metric("Selected Features", 50)
    c3.metric("Feature Reduction", "59.02%")

    show_figure(
        "xgboost_feature_reduction.png",
        "Original and selected feature counts"
    )

    st.divider()
    st.subheader("⭐ XGBoost Feature Importance")
    show_figure(
        "xgboost_feature_importance.png",
        "XGBoost feature importance"
    )

    selected_features_path = os.path.join(
        REPORTS_DIR, "selected_features.csv"
    )

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
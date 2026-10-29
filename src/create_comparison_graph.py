import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(
    REPORT_DIR,
    "final_model_comparison.csv"
)

# --------------------------------------------------
# LOAD FINAL MODEL COMPARISON
# --------------------------------------------------

print("=" * 65)
print("CREATING FINAL MODEL COMPARISON GRAPHS")
print("=" * 65)

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(
        f"Comparison CSV not found: {RESULTS_PATH}"
    )

results = pd.read_csv(RESULTS_PATH)

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

missing_columns = [
    column for column in required_columns
    if column not in results.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in CSV: {missing_columns}"
    )

results = results.dropna(subset=required_columns).copy()

if results.empty:
    raise ValueError("No valid model results found in the CSV.")

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for metric in metrics:
    results[metric] = pd.to_numeric(
        results[metric],
        errors="coerce"
    )

results = results.dropna(subset=metrics)

print(f"\nTotal model entries loaded: {len(results)}")
print("\nModel performance results:")
print(results[required_columns].to_string(index=False))

models = results["Model"].astype(str).tolist()
x = np.arange(len(models))

# --------------------------------------------------
# GRAPH 1: PERFORMANCE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(18, 8))

for metric in metrics:
    plt.plot(
        x,
        results[metric],
        marker="o",
        linewidth=2,
        markersize=5,
        label=metric
    )

plt.xticks(
    x,
    models,
    rotation=40,
    ha="right"
)

plt.ylabel("Performance (%)")
plt.xlabel("Model")
plt.title(
    "Comparative Analysis of Machine Learning Algorithms "
    "for Intrusion Detection"
)

plt.ylim(0, 105)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

performance_path = os.path.join(
    FIGURES_DIR,
    "final_model_performance_comparison.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nPerformance graph saved:\n{performance_path}")

# --------------------------------------------------
# GRAPH 2: ACCURACY COMPARISON
# --------------------------------------------------

accuracy_values = results["Accuracy"].to_numpy()
best_accuracy_index = int(np.argmax(accuracy_values))

plt.figure(figsize=(16, 8))

bars = plt.bar(x, accuracy_values)

bars[best_accuracy_index].set_edgecolor("black")
bars[best_accuracy_index].set_linewidth(2)

plt.xticks(
    x,
    models,
    rotation=40,
    ha="right"
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Accuracy Comparison of Intrusion Detection Models")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)

for index, value in enumerate(accuracy_values):
    plt.text(
        index,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

accuracy_path = os.path.join(
    FIGURES_DIR,
    "accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nAccuracy graph saved:\n{accuracy_path}")

# --------------------------------------------------
# GRAPH 3: GROUPED METRIC COMPARISON
# --------------------------------------------------

plt.figure(figsize=(19, 9))

bar_width = 0.19

for index, metric in enumerate(metrics):
    offset = (index - 1.5) * bar_width

    plt.bar(
        x + offset,
        results[metric],
        width=bar_width,
        label=metric
    )

plt.xticks(
    x,
    models,
    rotation=40,
    ha="right"
)

plt.ylabel("Performance (%)")
plt.xlabel("Model")
plt.title("Accuracy, Precision, Recall and F1-Score Comparison")
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

metrics_path = os.path.join(
    FIGURES_DIR,
    "all_metrics_comparison.png"
)

plt.savefig(
    metrics_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nAll-metrics graph saved:\n{metrics_path}")

# --------------------------------------------------
# GRAPH 4: XGBOOST FEATURE REDUCTION
# --------------------------------------------------

original_features = 122
selected_features = 50
features_removed = original_features - selected_features
reduction_percentage = (
    features_removed / original_features
) * 100

feature_labels = [
    "Original XGBoost",
    "Feature-Selected XGBoost"
]

feature_counts = [
    original_features,
    selected_features
]

plt.figure(figsize=(9, 6))

bars = plt.bar(feature_labels, feature_counts)

plt.ylabel("Number of Features")
plt.title("XGBoost Feature Reduction")
plt.ylim(0, 145)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, feature_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 3,
        str(value),
        ha="center",
        va="bottom"
    )

plt.figtext(
    0.5,
    0.01,
    f"Features removed: {features_removed} "
    f"({reduction_percentage:.2f}% reduction)",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

feature_path = os.path.join(
    FIGURES_DIR,
    "xgboost_feature_reduction.png"
)

plt.savefig(
    feature_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"\nFeature reduction graph saved:\n{feature_path}")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

best_accuracy_row = results.loc[
    results["Accuracy"].idxmax()
]

best_f1_row = results.loc[
    results["F1 Score"].idxmax()
]

print("\n" + "=" * 65)
print("GRAPH GENERATION COMPLETED")
print("=" * 65)

print(f"Model entries compared: {len(results)}")
print(
    f"Highest accuracy: {best_accuracy_row['Model']} "
    f"({best_accuracy_row['Accuracy']:.2f}%)"
)
print(
    f"Highest F1-score: {best_f1_row['Model']} "
    f"({best_f1_row['F1 Score']:.2f}%)"
)

print("\nGenerated graph files:")
print("1. final_model_performance_comparison.png")
print("2. accuracy_comparison.png")
print("3. all_metrics_comparison.png")
print("4. xgboost_feature_reduction.png")

print(f"\nAll files are located in:\n{FIGURES_DIR}")

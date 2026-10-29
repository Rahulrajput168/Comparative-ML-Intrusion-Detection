import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

FIGURES_DIR = os.path.join(
    REPORTS_DIR,
    "figures"
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------
# Load Evaluation Results
# --------------------------------------------------

results_path = os.path.join(
    REPORTS_DIR,
    "evaluation_results.csv"
)

df = pd.read_csv(results_path)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nEvaluation Results:")
print(df.to_string(index=False))


# --------------------------------------------------
# Metrics to Compare
# --------------------------------------------------

metrics = [
    ("Accuracy", "accuracy_comparison.png"),
    ("Precision", "precision_comparison.png"),
    ("Recall", "recall_comparison.png"),
    ("F1 Score", "f1_score_comparison.png")
]


# --------------------------------------------------
# Create Comparison Graphs
# --------------------------------------------------

for metric, filename in metrics:

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        df["Model"],
        df[metric]
    )

    plt.title(
        f"Model Comparison - {metric}",
        fontsize=14
    )

    plt.xlabel(
        "Machine Learning Model",
        fontsize=12
    )

    plt.ylabel(
        f"{metric} (%)",
        fontsize=12
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    # Add value above each bar
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    figure_path = os.path.join(
        FIGURES_DIR,
        filename
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    print(
        f"\n✅ {metric} graph saved:"
    )

    print(figure_path)


# --------------------------------------------------
# Find Best Models
# --------------------------------------------------

best_accuracy = df.loc[
    df["Accuracy"].idxmax()
]

best_precision = df.loc[
    df["Precision"].idxmax()
]

best_recall = df.loc[
    df["Recall"].idxmax()
]

best_f1 = df.loc[
    df["F1 Score"].idxmax()
]


# --------------------------------------------------
# Display Best Models
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST MODELS")
print("=" * 60)

print(
    f"\nBest Accuracy : "
    f"{best_accuracy['Model']} "
    f"({best_accuracy['Accuracy']:.2f}%)"
)

print(
    f"Best Precision: "
    f"{best_precision['Model']} "
    f"({best_precision['Precision']:.2f}%)"
)

print(
    f"Best Recall   : "
    f"{best_recall['Model']} "
    f"({best_recall['Recall']:.2f}%)"
)

print(
    f"Best F1 Score : "
    f"{best_f1['Model']} "
    f"({best_f1['F1 Score']:.2f}%)"
)

print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 60)
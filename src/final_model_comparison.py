import os
import pandas as pd

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

# --------------------------------------------------
# MODEL RESULTS
# Metrics are percentages; results are from existing reports.
# --------------------------------------------------

results = pd.DataFrame({

    "Model": [
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
    ],

    "Category": [
        "Traditional ML",
        "Traditional ML",
        "Traditional ML",
        "Traditional ML",
        "Traditional ML",
        "Gradient Boosting",
        "Gradient Boosting + Feature Selection",
        "Deep Learning - MLP",
        "Gradient Boosting - Optimized",
        "Deep Learning - Transformer"
    ],

    "Features": [
        122,
        122,
        122,
        122,
        122,
        122,
        50,
        122,
        122,
        122
    ],

    "Accuracy": [
        75.3815,
        78.5442,
        76.4771,
        77.0449,
        78.6950,
        79.5800,
        79.6487,
        80.0923,
        79.2362,
        80.3717
    ],

    "Precision": [
        91.7364,
        96.4776,
        96.7006,
        92.3375,
        92.5408,
        96.9100,
        96.9052,
        97.1522,
        96.8398,
        92.6196
    ],

    "Recall": [
        62.3705,
        64.6692,
        60.7496,
        65.0744,
        68.0589,
        66.2400,
        66.3680,
        66.9914,
        65.6666,
        71.1915
    ],

    "F1 Score": [
        74.2555,
        77.4341,
        74.6207,
        76.3450,
        78.4338,
        78.6900,
        78.7809,
        79.3008,
        78.2633,
        80.5040
    ]
})

# --------------------------------------------------
# DISPLAY COMPLETE COMPARISON
# --------------------------------------------------

print("\nFINAL MODEL COMPARISON:\n")
print(results.to_string(index=False))

# --------------------------------------------------
# BEST RESULT FOR EACH METRIC
# --------------------------------------------------

def print_best_result(metric, heading):
    best_row = results.loc[results[metric].idxmax()]

    print("\n" + "=" * 70)
    print(heading)
    print("=" * 70)
    print(f"Model : {best_row['Model']}")
    print(f"{metric}: {best_row[metric]:.2f}%")

print_best_result("Accuracy", "BEST ACCURACY")
print_best_result("Precision", "BEST PRECISION")
print_best_result("Recall", "BEST RECALL")
print_best_result("F1 Score", "BEST F1 SCORE")

# --------------------------------------------------
# XGBOOST FEATURE SELECTION COMPARISON
# --------------------------------------------------

xgb_original = results.loc[
    results["Model"] == "XGBoost"
].iloc[0]

xgb_selected = results.loc[
    results["Model"] == "Feature-Selected XGBoost"
].iloc[0]

print("\n" + "=" * 70)
print("XGBOOST FEATURE SELECTION IMPACT")
print("=" * 70)

print(f"Original features       : {int(xgb_original['Features'])}")
print(f"Selected features       : {int(xgb_selected['Features'])}")

feature_reduction = (
    xgb_original["Features"] - xgb_selected["Features"]
)

feature_reduction_percent = (
    feature_reduction / xgb_original["Features"]
) * 100

print(f"Features reduced        : {int(feature_reduction)}")
print(f"Feature reduction       : {feature_reduction_percent:.2f}%")

print(
    "Accuracy improvement   : "
    f"{xgb_selected['Accuracy'] - xgb_original['Accuracy']:.4f} "
    "percentage points"
)

print(
    "Recall improvement     : "
    f"{xgb_selected['Recall'] - xgb_original['Recall']:.4f} "
    "percentage points"
)

print(
    "F1-score improvement   : "
    f"{xgb_selected['F1 Score'] - xgb_original['F1 Score']:.4f} "
    "percentage points"
)

# --------------------------------------------------
# MLP RESULT
# --------------------------------------------------

mlp = results.loc[
    results["Model"] == "Deep Learning - MLP"
].iloc[0]

print("\n" + "=" * 70)
print("DEEP LEARNING MLP RESULT")
print("=" * 70)

print(f"Accuracy  : {mlp['Accuracy']:.4f}%")
print(f"Precision : {mlp['Precision']:.4f}%")
print(f"Recall    : {mlp['Recall']:.4f}%")
print(f"F1 Score  : {mlp['F1 Score']:.4f}%")

# --------------------------------------------------
# TRANSFORMER RESULT
# --------------------------------------------------

transformer = results.loc[
    results["Model"] == "Tabular Transformer"
].iloc[0]

print("\n" + "=" * 70)
print("TABULAR TRANSFORMER RESULT")
print("=" * 70)

print(f"Accuracy  : {transformer['Accuracy']:.4f}%")
print(f"Precision : {transformer['Precision']:.4f}%")
print(f"Recall    : {transformer['Recall']:.4f}%")
print(f"F1 Score  : {transformer['F1 Score']:.4f}%")

# --------------------------------------------------
# SAVE FINAL COMPARISON CSV
# --------------------------------------------------

output_path = os.path.join(
    REPORT_DIR,
    "final_model_comparison.csv"
)

results.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("FINAL COMPARISON SAVED SUCCESSFULLY")
print("=" * 70)
print(f"File: {output_path}")
print(f"Total model entries: {len(results)}")

print("\nNote: Metrics are combined from existing report files.")
print("Verify common test data and evaluation settings before")
print("treating the rankings as a fully controlled comparison.")

import os
import sys

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Project root ko Python path me add karo
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from dataset.load_dataset import load_data


def preprocess_data():

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------

    train_df, test_df = load_data()

    print("\nOriginal Training Shape:", train_df.shape)
    print("Original Testing Shape :", test_df.shape)

    # --------------------------------------------------
    # 2. Remove Difficulty Column
    # --------------------------------------------------

    train_df = train_df.drop(columns=["difficulty"])
    test_df = test_df.drop(columns=["difficulty"])

    # --------------------------------------------------
    # 3. Convert Target into Binary Classification
    #
    # 0 = Normal
    # 1 = Attack
    # --------------------------------------------------

    train_df["label"] = train_df["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    test_df["label"] = test_df["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # --------------------------------------------------
    # 4. Separate Features and Target
    # --------------------------------------------------

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # --------------------------------------------------
    # 5. Find Categorical Columns
    # --------------------------------------------------

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\nCategorical Columns:")
    print(categorical_columns)

    # --------------------------------------------------
    # 6. One-Hot Encoding
    # --------------------------------------------------

    X_train = pd.get_dummies(
        X_train,
        columns=categorical_columns,
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_columns,
        dtype=int
    )

    # --------------------------------------------------
    # 7. Make Train and Test Columns Identical
    # --------------------------------------------------

    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0
    )

    # --------------------------------------------------
    # 8. Feature Scaling
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

    X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns
)

    # --------------------------------------------------
    # 9. Create Processed Data Folder
    # --------------------------------------------------

    processed_dir = os.path.join(
        PROJECT_ROOT,
        "processed_data"
    )

    os.makedirs(processed_dir, exist_ok=True)

    # --------------------------------------------------
    # 10. Save Processed Data
    # --------------------------------------------------

    pd.DataFrame(X_train).to_csv(
        os.path.join(processed_dir, "X_train.csv"),
        index=False
    )

    pd.DataFrame(X_test).to_csv(
        os.path.join(processed_dir, "X_test.csv"),
        index=False
    )

    pd.DataFrame(y_train).to_csv(
        os.path.join(processed_dir, "y_train.csv"),
        index=False
    )

    pd.DataFrame(y_test).to_csv(
        os.path.join(processed_dir, "y_test.csv"),
        index=False
    )

    # Save scaler
    joblib.dump(
        scaler,
        os.path.join(processed_dir, "scaler.pkl")
    )

    # Save feature names
    joblib.dump(
        X_train.shape[1],
        os.path.join(processed_dir, "feature_count.pkl")
    )

    # --------------------------------------------------
    # 11. Display Results
    # --------------------------------------------------

    print("\nAfter Preprocessing:")

    print("X_train Shape:", X_train.shape)
    print("X_test Shape :", X_test.shape)
    print("y_train Shape:", y_train.shape)
    print("y_test Shape :", y_test.shape)

    print("\nTarget Distribution - Training:")
    print(y_train.value_counts())

    print("\nTarget Distribution - Testing:")
    print(y_test.value_counts())

    print("\nTarget Meaning:")
    print("0 = Normal Traffic")
    print("1 = Attack Traffic")

    print("\nProcessed files saved in:")
    print(processed_dir)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_data()
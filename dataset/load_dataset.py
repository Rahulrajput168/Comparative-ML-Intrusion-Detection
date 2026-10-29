import pandas as pd
from dataset.config import TRAIN_DATA_PATH, TEST_DATA_PATH, COLUMN_NAMES

def load_data():
    """
    Load training and testing datasets.
    """

    train_df = pd.read_csv(
        TRAIN_DATA_PATH,
        names=COLUMN_NAMES
    )

    test_df = pd.read_csv(
        TEST_DATA_PATH,
        names=COLUMN_NAMES
    )

    return train_df, test_df


if __name__ == "__main__":

    train_df, test_df = load_data()

    print("=" * 60)
    print("Training Dataset Loaded Successfully")
    print("=" * 60)

    print(train_df.head())

    print("\nTraining Shape :", train_df.shape)

    print("\n")

    print("=" * 60)
    print("Testing Dataset Loaded Successfully")
    print("=" * 60)

    print(test_df.head())

    print("\nTesting Shape :", test_df.shape)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from dataset.load_dataset import load_data

# Load Dataset
train_df, test_df = load_data()

print("=" * 60)
print("NSL-KDD DATASET ANALYSIS")
print("=" * 60)

# Shape
print("\n1. Dataset Shape")
print("-" * 60)
print("Training Dataset :", train_df.shape)
print("Testing Dataset  :", test_df.shape)

# Columns
print("\n2. Column Names")
print("-" * 60)
print(train_df.columns.tolist())

# Data Types
print("\n3. Data Types")
print("-" * 60)
print(train_df.dtypes)

# Missing Values
print("\n4. Missing Values")
print("-" * 60)
print(train_df.isnull().sum())

# Duplicate Values
print("\n5. Duplicate Rows")
print("-" * 60)
print(train_df.duplicated().sum())

# Class Distribution
print("\n6. Attack Class Distribution")
print("-" * 60)
print(train_df["label"].value_counts())
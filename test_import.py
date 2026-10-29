from dataset.load_dataset import load_data

train_df, test_df = load_data()

print("Training Shape:", train_df.shape)
print("Testing Shape:", test_df.shape)
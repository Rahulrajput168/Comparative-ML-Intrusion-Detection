import os
import sys
import matplotlib.pyplot as plt

# Project root ko Python path me add karo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.load_dataset import load_data

# Load Dataset
train_df, test_df = load_data()

# Create reports/figures folder if it doesn't exist
os.makedirs("reports/figures", exist_ok=True)

# Get Top 10 Attack Classes
attack_counts = train_df["label"].value_counts().head(10)

# Create Figure
plt.figure(figsize=(10, 6))

# Plot Bar Chart
attack_counts.plot(kind="bar", color="skyblue", edgecolor="black")

# Title and Labels
plt.title("Top 10 Attack Classes in NSL-KDD Dataset", fontsize=14)
plt.xlabel("Attack Type", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Rotate X-axis Labels
plt.xticks(rotation=45)

# Adjust Layout
plt.tight_layout()

# Save Figure
plt.savefig("reports/figures/attack_distribution.png", dpi=300)

# Show Figure
plt.show()

print("=" * 60)
print("✅ Attack Distribution Graph Generated Successfully!")
print("📁 Saved at : reports/figures/attack_distribution.png")
print("=" * 60)
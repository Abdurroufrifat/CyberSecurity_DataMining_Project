import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# Load Dataset
# -----------------------------
root = Path(__file__).resolve().parent.parent
dataset_path = root / "dataset" / "preprocessed_dataset.csv"

print("Loading dataset...")

# Use 300,000 rows only for faster EDA
df = pd.read_csv(dataset_path, low_memory=False, nrows=300000)

# Create figures folder
figures_path = root / "figures"
figures_path.mkdir(exist_ok=True)

sns.set_style("whitegrid")

# =============================
# 1. Attack Distribution
# =============================
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="Binary_Label")
plt.title("Attack Distribution")
plt.xlabel("Binary Label (0 = Benign, 1 = Attack)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(figures_path / "attack_distribution.png", dpi=300)
plt.close()

# =============================
# 2. Traffic Pie Chart
# =============================
counts = df["Binary_Label"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=["Benign", "Attack"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Traffic Distribution")
plt.tight_layout()
plt.savefig(figures_path / "traffic_pie_chart.png", dpi=300)
plt.close()

# =============================
# 3. Correlation Heatmap
# =============================
numeric_df = df.select_dtypes(include=["number"])

corr_with_label = numeric_df.corr()["Binary_Label"].abs().sort_values(ascending=False)

top_features = corr_with_label.head(16).index

corr = numeric_df[top_features].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Top Features")
plt.tight_layout()
plt.savefig(figures_path / "correlation_heatmap.png", dpi=300)
plt.close()

print("\nEDA Completed Successfully!")
print("Figures saved in:", figures_path)
print("Generated files:")
print("1. attack_distribution.png")
print("2. traffic_pie_chart.png")
print("3. correlation_heatmap.png")
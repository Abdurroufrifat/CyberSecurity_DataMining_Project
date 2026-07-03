import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------------
# Load Dataset
# ----------------------------
root = Path(__file__).resolve().parent.parent

dataset_path = root / "dataset" / "preprocessed_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(dataset_path, low_memory=False)

# ----------------------------
# Features and Target
# ----------------------------
X = df.drop(["Label", "Binary_Label"], axis=1)
y = df["Binary_Label"]

# Keep only numeric features
X = X.select_dtypes(include=["number"])

print("Feature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

# ----------------------------
# Feature Standardization
# ----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
models_path = root / "models"
models_path.mkdir(exist_ok=True)

joblib.dump(scaler, models_path / "scaler.pkl")

# Save processed data
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test),
            models_path / "processed_data.pkl")

print("\nFeature Engineering Completed Successfully!")
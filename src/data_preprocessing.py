import pandas as pd
import numpy as np
from pathlib import Path


def preprocess_dataset(input_file, output_file):
    print("Loading combined dataset...")
    df = pd.read_csv(input_file, low_memory=False, encoding="utf-8")

    print("Original shape:", df.shape)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Replace infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove non-useful features if they exist
    columns_to_drop = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Convert labels: BENIGN = 0, Attack = 1
    df["Binary_Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    print("After preprocessing shape:", df.shape)
    print("\nBinary label distribution:")
    print(df["Binary_Label"].value_counts())

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nPreprocessed dataset saved to: {output_file}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    input_file = root / "dataset" / "combined_dataset.csv"
    output_file = root / "dataset" / "preprocessed_dataset.csv"

    preprocess_dataset(input_file, output_file)
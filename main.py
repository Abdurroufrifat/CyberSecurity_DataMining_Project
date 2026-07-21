import sys
import io
import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import pandas as pd


def load_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Load all CSV files from the dataset directory."""

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    csv_files = sorted(
        path for path in dataset_dir.glob("*.csv")
        if path.name.lower() not in {"combined_dataset.csv", "preprocessed_dataset.csv"}
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in dataset directory: {dataset_dir} (excluding combined_dataset.csv and preprocessed_dataset.csv)"
        )

    print("CSV Files Found:")
    for path in csv_files:
        print(path.name)

    dataframes = []

    for path in csv_files:
        print(f"Loading {path.name}...")
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        dataframes.append(df)

    dataset = pd.concat(dataframes, ignore_index=True)

    # Remove extra spaces from column names and drop duplicate columns
    dataset.columns = dataset.columns.str.strip()
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    return dataset


def main() -> int:
    root = Path(__file__).resolve().parent
    dataset_dir = root / "dataset"

    try:
        dataset = load_dataset(dataset_dir)

    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    except pd.errors.EmptyDataError as exc:
        print(f"Error reading CSV file: {exc}")
        return 2

    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 3

    print("\nDataset Loaded Successfully!")
    print("Shape:", dataset.shape)

    print("\nFirst 5 Rows")
    print(dataset.head())

    print("\nInformation")
    dataset.info()

    print("\nStatistics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        print("Numeric summary:")
        print(dataset.describe())
        print("\nAll columns summary:")
        print(dataset.describe(include="all", datetime_is_numeric=True))

    print("\nColumns")
    print(dataset.columns.tolist())

    print("\nAttack Types")

    if "Label" in dataset.columns:
        print(dataset["Label"].value_counts())
    else:
        print("Target column not found. Please check column names.")

    # Save combined dataset outside the raw dataset directory to prevent reloading it
    output_path = root / "combined_dataset.csv"
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nCombined dataset saved successfully at: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
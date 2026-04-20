import pandas as pd

def load_data(path="data/raw/healthcare_dataset.csv"):
    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nMissing values:\n", df.isnull().sum())

    return df


def main():
    print("  Starting Healthcare ML Pipeline...\n")

    df = load_data()

    print("\n Data loaded successfully")
    print(df.head())


if __name__ == "__main__":
    main()
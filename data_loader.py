import pandas as pd

def load_data(train_path: str, test_path: str):
    print("Loading training and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("\nTrain Columns:", train_df.columns)
    print("Train Shape:", train_df.shape)
    print("\nSample data:")
    print(train_df.head())

    return train_df, test_df

if __name__ == "__main__":
    train_data, test_data = load_data("data/train.csv", "data/test.csv")

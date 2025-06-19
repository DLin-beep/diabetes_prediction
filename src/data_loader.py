import pandas as pd
import os

def load_diabetes_data(data_path=None):
    """
    Load the diabetes dataset from the data directory.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes.csv')
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_diabetes_data()
    print(df.head()) 
import os
import pandas as pd

def register_data():
    file_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please upload it.")

    df = pd.read_csv(file_path)
    print("Validation Passed: Dataset loaded successfully.")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

if __name__ == "__main__":
    register_data()


import os  # Provides utilities for file path validation and file system operations
import pandas as pd  # Used for loading and processing structured datasets

def register_data():
    # Define the path to the source dataset
    file_path = "tourism_project/data/tourism.csv"

    # Validate dataset availability before attempting to load it
    if not os.path.exists(file_path):
        # Raise an exception with a descriptive message if the dataset is missing
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please upload it.")

    # Load the CSV dataset into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Confirm successful dataset loading and validation
    print("Validation Passed: Dataset loaded successfully.")

    # Display dataset dimensions for basic data verification
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

# Execute dataset registration only when the script is run directly
if __name__ == "__main__":
    register_data()

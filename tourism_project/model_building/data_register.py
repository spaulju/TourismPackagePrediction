import pandas as pd
import os

def register_data():
    """
    Reads the raw dataset from the data folder, validates that all expected 
    features are present, and outputs a structural summary.
    """
    file_path = "tourism_project/data/tourism.csv"
    
    # Check if the file exists in the correct directory
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please upload it to the 'data' folder.")
    
    df = pd.read_csv(file_path)
    
    # Define the expected columns based on the data dictionary
    expected_cols = [
        'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]
       
    # Validate column presence
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: The following expected columns are missing: {missing_cols}")
    else:
        print("SUCCESS: All expected columns are present in the dataset.")
        
    # Print dataset summary
    print("\nDataset Info:")
    print(df.info())
    print("\nData registration and validation complete.")

if __name__ == "__main__":
    register_data()

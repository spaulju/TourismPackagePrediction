import os
import pandas as pd

def register_data():
    """
    Reads the raw dataset from the repository data folder, checks that all 
    expected columns are present, and prints a dataset summary.
    """
    file_path = "tourism_project/data/tourism.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please upload tourism.csv to tourism_project/data/.")
    
    df = pd.read_csv(file_path)
    
    # Expected columns as per project Data Dictionary
    expected_cols = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]
       
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    else:
        print("Validation Passed: All expected columns are present.")
        
    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print(df.info())

if __name__ == "__main__":
    register_data()

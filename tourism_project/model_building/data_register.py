import pandas as pd
import os

def register_data():
    file_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please ensure it is uploaded.")
    
    df = pd.read_csv(file_path)
    
    expected_cols = [
        'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
    ]
       
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    else:
        print("All expected columns are present.")
        
    print(df.info())
    print("Data registration complete.")

if __name__ == "__main__":
    register_data()

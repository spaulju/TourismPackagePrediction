import pandas as pd
from sklearn.model_selection import train_test_split

def prep_data():
    """
    Cleans the raw dataset, handles missing values, removes useless identifiers, 
    and splits the data into training and testing artifacts.
    """
    file_path = "tourism_project/data/tourism.csv"
    df = pd.read_csv(file_path)
    
    # Drop CustomerID as it has no predictive value
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
        
    # Drop hidden index columns (fixes the Streamlit 'Unnamed: 0' error)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        
    # Python 3.11 / Pandas 2.x compatible missing value imputation
    # Forward fill then backward fill to ensure no NaNs remain
    df = df.ffill().bfill()
    
    # Separate features (X) and target (y)
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    
    # Split the dataset (80% training, 20% testing)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the splits locally to be picked up by the training script
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    
    print("Data cleaning, imputation, and train-test splitting complete.")

if __name__ == "__main__":
    prep_data()

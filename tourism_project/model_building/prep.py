import pandas as pd
from sklearn.model_selection import train_test_split

def prep_data():
    file_path = "tourism_project/data/tourism.csv"
    df = pd.read_csv(file_path)
    
    # Drop CustomerID if it exists
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
        
    # Basic missing value handling
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    
    # Split into train and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save locally to be used as workflow artifacts
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    print("Data preparation and splitting complete.")

if __name__ == "__main__":
    prep_data()

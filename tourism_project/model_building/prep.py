import pandas as pd
from sklearn.model_selection import train_test_split

def prep_data():
    """
    Loads raw data, drops identifier/index columns, cleans data entry typos,
    splits into train/test sets, and saves them locally as CSV artifacts.
    """
    file_path = "tourism_project/data/tourism.csv"
    df = pd.read_csv(file_path)
    
    # Remove unnecessary index and identifier columns
    cols_to_drop = ['Unnamed: 0', 'CustomerID']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Clean standard dataset typos in categorical columns
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
    if 'MaritalStatus' in df.columns:
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
        
    # Handle missing values using median for numeric and mode for categorical
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Separate features and target variable
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    
    # Split dataset into 80% train and 20% test
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save splits locally for the next GitHub Actions job
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)
    
    print(f"Data preparation complete. Features used ({Xtrain.shape[1]}): {list(Xtrain.columns)}")

if __name__ == "__main__":
    prep_data()


import pandas as pd  # Used for data manipulation, cleaning, and CSV file operations
from sklearn.model_selection import train_test_split  # Used to split the dataset into training and testing sets

def prep_data():
    # Load the source dataset into a Pandas DataFrame
    df = pd.read_csv("tourism_project/data/tourism.csv")

    # Define unnecessary columns to exclude from model training
    cols_to_drop = ['Unnamed: 0', 'CustomerID']

    # Remove specified columns only if they exist in the dataset
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Standardize inconsistent categorical values caused by data entry errors
    if 'Gender' in df.columns:
        # Correct inconsistent gender labels
        df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

    if 'MaritalStatus' in df.columns:
        # Standardize marital status labels for consistent categorization
        df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

    # Handle missing values based on the data type of each column
    for col in df.columns:
        if df[col].dtype == 'object':
            # Replace missing categorical values with the most frequent value
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # Replace missing numerical values with the column median
            df[col] = df[col].fillna(df[col].median())

    # Separate independent features from the target variable
    X = df.drop('ProdTaken', axis=1)

    # Define the target variable for model prediction
    y = df['ProdTaken']

    # Split data into 80% training and 20% testing sets
    # Use a fixed random state for reproducibility and stratification to preserve target class distribution
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save training features to a CSV file without row indices
    Xtrain.to_csv("Xtrain.csv", index=False)

    # Save testing features to a CSV file without row indices
    Xtest.to_csv("Xtest.csv", index=False)

    # Save training target labels to a CSV file without row indices
    ytrain.to_csv("ytrain.csv", index=False)

    # Save testing target labels to a CSV file without row indices
    ytest.to_csv("ytest.csv", index=False)

    # Confirm successful completion of the data preparation process
    print("Data preparation complete.")

# Execute data preparation only when the script is run directly
if __name__ == "__main__":
    prep_data()

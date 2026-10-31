
import os  # Provides utilities for directory creation and file system operations
import joblib  # Used for serializing and saving the trained machine learning model
import pandas as pd  # Used for loading and manipulating training and testing datasets
import mlflow  # Used for experiment tracking and logging model performance metrics
from sklearn.compose import make_column_transformer  # Applies preprocessing transformations to specific feature columns
from sklearn.pipeline import make_pipeline  # Combines preprocessing and model training into a single pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Used for numerical feature scaling and categorical feature encoding
from sklearn.impute import SimpleImputer  # Handles missing values in numerical and categorical features
from sklearn.ensemble import RandomForestClassifier  # Implements the Random Forest classification algorithm
from sklearn.model_selection import GridSearchCV  # Performs hyperparameter tuning using cross-validation
from sklearn.metrics import accuracy_score  # Evaluates model prediction accuracy

def train_model():
    # Load the preprocessed training and testing feature datasets
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")

    # Load target labels and convert them into one-dimensional arrays
    ytrain = pd.read_csv("ytrain.csv").values.ravel()
    ytest = pd.read_csv("ytest.csv").values.ravel()

    # Identify categorical feature columns based on object data types
    categorical_cols = Xtrain.select_dtypes(include=['object']).columns.tolist()

    # Identify numerical feature columns by excluding object data types
    numeric_cols = Xtrain.select_dtypes(exclude=['object']).columns.tolist()

    # Define preprocessing transformations for numerical and categorical features
    preprocessor = make_column_transformer(
        # Impute missing numerical values using the median and standardize feature values
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_cols),

        # Impute missing categorical values using the most frequent value and apply one-hot encoding
        # Ignore unseen categories during prediction to prevent transformation errors
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical_cols)
    )

    # Create an end-to-end pipeline combining feature preprocessing and Random Forest classification
    model_pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))

    # Define the hyperparameter search space for Random Forest model optimization
    param_grid = {'randomforestclassifier__n_estimators': [50, 100], 'randomforestclassifier__max_depth': [None, 10]}

    # Configure the MLflow tracking server for experiment logging
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set the MLflow experiment name for organizing training runs
    mlflow.set_experiment("Wellness_Tourism_Prediction")

    # Start an MLflow run to track model training and evaluation results
    with mlflow.start_run():

        # Configure grid search with 3-fold cross-validation and accuracy-based scoring
        # Use all available CPU cores for parallel hyperparameter evaluation
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

        # Train models across all hyperparameter combinations using the training dataset
        grid_search.fit(Xtrain, ytrain)

        # Retrieve the best-performing model selected through cross-validation
        best_model = grid_search.best_estimator_

        # Generate predictions on the test dataset and calculate classification accuracy
        acc = accuracy_score(ytest, best_model.predict(Xtest))

        # Log the test accuracy metric to the active MLflow experiment run
        mlflow.log_metric("test_accuracy", acc)

        # Display the model's test accuracy rounded to four decimal places
        print(f"Test Accuracy: {acc:.4f}")

        # Create the deployment directory if it does not already exist
        os.makedirs("tourism_project/deployment", exist_ok=True)

        # Serialize and save the best-performing model for future deployment and inference
        joblib.dump(best_model, "tourism_project/deployment/best_model.pkl")

# Execute model training only when the script is run directly
if __name__ == "__main__":
    train_model()

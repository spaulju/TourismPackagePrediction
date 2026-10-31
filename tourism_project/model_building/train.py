import pandas as pd
import mlflow
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train_model():
    """
    Loads data splits, builds a preprocessing and classification pipeline,
    tunes hyperparameters, logs metrics to MLflow, and saves the best model.
    """
    # Load artifacts from the prep step
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").values.ravel()
    ytest = pd.read_csv("ytest.csv").values.ravel()
    
    # Identify column types for preprocessing
    categorical_cols = Xtrain.select_dtypes(include=['object']).columns
    numeric_cols = Xtrain.select_dtypes(exclude=['object']).columns

    # Pipeline for numeric features: Impute missing -> Standardize
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Pipeline for categorical features: Impute missing -> One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Append classifier to preprocessing pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
                            
    # Define hyperparameter grid for tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10]
    }
    
    # Configure MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Wellness_Tourism_Prediction")
    
    # Start MLflow run
    with mlflow.start_run():
        # Perform grid search cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)
        
        # Extract best model and log parameters
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate model on test set
        predictions = best_model.predict(Xtest)
        acc = accuracy_score(ytest, predictions)
        mlflow.log_metric("accuracy", acc)
        
        print(f"GridSearch Best Parameters: {grid_search.best_params_}")
        print(f"Model Accuracy on Test Data: {acc:.4f}")
        
        # Save the optimized model to the deployment folder
        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, "tourism_project/deployment/best_model.pkl")
        print("Best model successfully committed to tourism_project/deployment/best_model.pkl")

if __name__ == "__main__":
    train_model()

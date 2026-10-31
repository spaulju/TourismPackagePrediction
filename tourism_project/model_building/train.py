import os
import joblib
import pandas as pd
import mlflow
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    """
    Loads train/test splits, tunes a Random Forest pipeline using GridSearchCV,
    logs parameters and metrics to MLflow, and saves the best model for deployment.
    """
    # Load split data artifacts
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").values.ravel()
    ytest = pd.read_csv("ytest.csv").values.ravel()
    
    # Identify numerical and categorical columns
    categorical_cols = Xtrain.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = Xtrain.select_dtypes(exclude=['object']).columns.tolist()

    # Build preprocessing pipelines
    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    )

    preprocessor = make_column_transformer(
        (num_pipeline, numeric_cols),
        (cat_pipeline, categorical_cols)
    )

    # Define full modeling pipeline
    model_pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=42)
    )
                            
    # Define hyperparameter grid
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100],
        'randomforestclassifier__max_depth': [None, 10, 15],
        'randomforestclassifier__min_samples_split': [2, 5]
    }
    
    # Configure MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Wellness_Tourism_Prediction")
    
    with mlflow.start_run():
        grid_search = GridSearchCV(
            model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(Xtrain, ytrain)
        
        best_model = grid_search.best_estimator_
        
        # Log best hyperparameters
        mlflow.log_params(grid_search.best_params_)
        
        # Evaluate on test data
        predictions = best_model.predict(Xtest)
        acc = accuracy_score(ytest, predictions)
        mlflow.log_metric("test_accuracy", acc)
        
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Test Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", classification_report(ytest, predictions))
        
        # Save the best model to the deployment directory
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_save_path = "tourism_project/deployment/best_model.pkl"
        joblib.dump(best_model, model_save_path)
        print(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    train_model()

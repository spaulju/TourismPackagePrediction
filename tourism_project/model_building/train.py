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
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").values.ravel()
    ytest = pd.read_csv("ytest.csv").values.ravel()
    
    categorical_cols = Xtrain.select_dtypes(include=['object']).columns
    numeric_cols = Xtrain.select_dtypes(exclude=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
                            
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10]
    }
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Wellness_Tourism_Prediction")
    
    with mlflow.start_run():
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)
        
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        
        predictions = best_model.predict(Xtest)
        acc = accuracy_score(ytest, predictions)
        mlflow.log_metric("accuracy", acc)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Model Accuracy on Test Set: {acc}")
        
        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, "tourism_project/deployment/best_model.pkl")
        print("Best model committed to tourism_project/deployment/best_model.pkl")

if __name__ == "__main__":
    train_model()

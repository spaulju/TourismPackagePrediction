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
from sklearn.metrics import accuracy_score

def train_model():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").values.ravel()
    ytest = pd.read_csv("ytest.csv").values.ravel()

    categorical_cols = Xtrain.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = Xtrain.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_cols),
        (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical_cols)
    )

    model_pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))

    param_grid = {'randomforestclassifier__n_estimators': [50, 100], 'randomforestclassifier__max_depth': [None, 10]}

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Wellness_Tourism_Prediction")

    with mlflow.start_run():
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)
        best_model = grid_search.best_estimator_

        acc = accuracy_score(ytest, best_model.predict(Xtest))
        mlflow.log_metric("test_accuracy", acc)
        print(f"Test Accuracy: {acc:.4f}")

        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, "tourism_project/deployment/best_model.pkl")

if __name__ == "__main__":
    train_model()

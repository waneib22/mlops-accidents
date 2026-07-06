
import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(joblib.__version__)

def train():
    #Tracking :
    import mlflow
    mlflow.set_tracking_uri("http://localhost:8080") #url local
    #mlflow.set_tracking_uri("http://mlflow:8080")  # via reseau docker

    # Set experiment 
    mlflow.set_experiment("Prediction_Accidents")

    X_train = pd.read_csv('data/preprocessed/X_train.csv')
    X_test  = pd.read_csv('data/preprocessed/X_test.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')
    y_test  = pd.read_csv('data/preprocessed/y_test.csv')

    y_train = np.ravel(y_train) # np.ravel() sert à transformer un label 2D en 1D pour sklearn
    y_test = np.ravel(y_test)

    with mlflow.start_run(run_name="RandomForest_Baseline_Mélanie"):

        # Params à tracker
        n_estimators = 100
        mlflow.log_params({
                "n_estimators": n_estimators,
                "model": "RandomForest",
                "n_jobs": -1
            })

        # Model
        rf_classifier = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,       # Limite la profondeur pour alléger le modèle (car pas assez de place sur mon disque dur)
            n_jobs=-1,
            random_state=42
        )

        rf_classifier.fit(X_train, y_train)

        # Score
        y_pred = rf_classifier.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
}
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Save model , pas obligatoire car on sauvegarde mtn sur mlflow
        joblib.dump(rf_classifier, "./src/models/trained_model.joblib")

        # Log model MLflow + Model registry
        mlflow.sklearn.log_model(
            sk_model=rf_classifier,
            name="model RF",
            registered_model_name="Modele Random Forest") 

        print("Model trained, logged and registered with MLflow")

    return rf_classifier


if __name__ == "__main__":
    train()
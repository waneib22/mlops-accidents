import joblib
import pandas as pd
import numpy as np
import json
import logging
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_test_data(input_folderpath="data/preprocessed"):
    """Load preprocessed test features and labels."""
    X_test = pd.read_csv(f"{input_folderpath}/X_test.csv")
    y_test = pd.read_csv(f"{input_folderpath}/y_test.csv")
    y_test = np.ravel(y_test)
    return X_test, y_test


def load_model(model_path="./src/models/trained_model.joblib"):
    """Load the trained model from disk."""
    return joblib.load(model_path)


def evaluate(model, X_test, y_test):
    """Compute evaluation metrics for the given model and test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return metrics, cm, report


def save_metrics(metrics, output_path="metrics/evaluation_metrics.json"):
    """Save metrics to a JSON file for later (ex:CI, DVC metrics)."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)


def main(
    model_path="./src/models/trained_model.joblib",
    input_folderpath="data/preprocessed",
    metrics_output_path="metrics/evaluation_metrics.json",
    log_to_mlflow=True,
):
    logger = logging.getLogger(__name__)
    logger.info("Loading test data and trained model")

    X_test, y_test = load_test_data(input_folderpath)
    model = load_model(model_path)

    logger.info("Evaluating model")
    metrics, cm, report = evaluate(model, X_test, y_test)

    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(report)

    save_metrics(metrics, metrics_output_path)
    logger.info(f"Metrics saved to {metrics_output_path}")

    if log_to_mlflow:
        mlflow.set_tracking_uri("http://localhost:8080")
        mlflow.set_experiment("Prediction_Accidents")
        with mlflow.start_run(run_name="Evaluation_RandomForest_Melanie"):
            for name, value in metrics.items():
                mlflow.log_metric(f"eval_{name}", value)
            mlflow.log_artifact(metrics_output_path)
        logger.info("Evaluation metrics logged to MLflow")

    return metrics


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

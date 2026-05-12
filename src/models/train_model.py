"""Script d'entraînement du modèle RandomForestClassifier.

Lit X_train/y_train depuis data/processed/, entraîne le modèle, évalue
l'accuracy sur X_test/y_test, puis sérialise le modèle avec joblib.
"""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn import ensemble

from src.config.config import DATA_PROCESSED_DIR, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train():
    logger.info("Chargement des données...")
    X_train = pd.read_csv(DATA_PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_test.csv")
    y_train = np.ravel(pd.read_csv(DATA_PROCESSED_DIR / "y_train.csv"))
    y_test = np.ravel(pd.read_csv(DATA_PROCESSED_DIR / "y_test.csv"))

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    logger.info("Entraînement du RandomForestClassifier...")
    model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    logger.info(f"Accuracy sur le test set : {score:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Modèle sauvegardé dans {MODEL_PATH}")

    return model, score


if __name__ == "__main__":
    train()

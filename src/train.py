import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Chargement des données
DOSSIER_PROCESSED = Path("data/processed")
df = pd.read_csv(DOSSIER_PROCESSED / "accidents_2024_processed.csv")

y = df["grav"]
X = df.drop(columns=["grav"])
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nRésultats :")
print(classification_report(y_test, y_pred))

DOSSIER_MODELS = Path("models")
DOSSIER_MODELS.mkdir(parents=True, exist_ok=True)

joblib.dump(model, DOSSIER_MODELS / "model.pkl")
joblib.dump(X.columns.tolist(), DOSSIER_MODELS / "model_columns.pkl")

print("\nModèle sauvegardé")
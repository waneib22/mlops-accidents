print("DEBUT DU SCRIPT")

import mlflow
from mlflow import MlflowClient

print("mlflow importe")

mlflow.set_tracking_uri("http://localhost:8080")
print("Tracking URI:", mlflow.get_tracking_uri())

client = MlflowClient()
print("Client cree, recherche des modeles...")

models = client.search_registered_models()
print("Nombre de modeles trouves:", len(models))

for m in models:
    print("Modele:", repr(m.name))

print("FIN DU SCRIPT")
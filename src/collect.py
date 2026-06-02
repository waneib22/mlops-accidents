import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ressources = {
    "caracteristiques_2024": "83f0fb0e-e0ef-47fe-93dd-9aaee851674a",
    "lieux_2024": "228b3cda-fdfb-4677-bd54-ab2107028d2d",
    "vehicules_2024": "fd30513c-6b11-4a56-b6dc-5ac87728794b"
    #"usagers_2024": "f57b1f58-386d-4048-8f78-2ebe435df868"
    }

for nom_fichier, rid in ressources.items():
    url = f"https://tabular-api.data.gouv.fr/api/resources/{rid}/data/"
    lignes = []
    max_pages = 5
    page = 0
    while url and page < max_pages:
        page += 1
        print(f"Récupération page {page} :", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        contenu = response.json()
        lignes.extend(contenu["data"])
        url = contenu["links"]["next"]

    df = pd.DataFrame(lignes)
    path_sortie = RAW_DIR / f"{nom_fichier}.csv"
    df.to_csv(path_sortie, index=False)
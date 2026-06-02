import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

caract = pd.read_csv(RAW_DIR / "caracteristiques_2024.csv")
lieux = pd.read_csv(RAW_DIR / "lieux_2024.csv")
vehicules = pd.read_csv(RAW_DIR / "vehicules_2024.csv")
usagers = pd.read_csv(RAW_DIR / "usagers_2024.csv", sep=";")

print(caract.columns)
print(lieux.columns)
print(vehicules.columns)
print(usagers.columns)

df = pd.merge(caract, lieux, on="Num_Acc")
df = pd.merge(df, vehicules, on="Num_Acc")
df = pd.merge(df, usagers, on="Num_Acc")

print(df.head())
print(df.shape)
print("Nombre de lignes :", len(df))
print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

#NETTOYAGE

df = df.drop(["__id_x", "__id_y", "__id","lartpc","v2","occutc"], axis=1)
df = df.fillna("inconnu")


DOSSIER_PROCESSED = Path("data/processed")
DOSSIER_PROCESSED.mkdir(parents=True, exist_ok=True)
Path_sortie = DOSSIER_PROCESSED / "accidents_2024_processed.csv"
df.to_csv(Path_sortie, index=False)

print(df.shape)
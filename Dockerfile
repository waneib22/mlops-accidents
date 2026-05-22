
# Image Python - même version que pyproject.toml
FROM python:3.13

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copier requirements en premier (optimise le cache Docker)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'API principale
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
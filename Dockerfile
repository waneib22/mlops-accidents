
# Python base (stable recommandé, 3.11 ou 3.12 > 3.13 pour compatibilité)
FROM python:3.11-slim

WORKDIR /app

# Installer uv
RUN pip install --no-cache-dir uv

# Copier uniquement les fichiers de dépendances d'abord (cache Docker)
COPY requirements.txt ./

# Installer les dépendances avec uv (plus rapide que pip)
RUN uv pip install --system -r requirements.txt

# Copier le reste du projet
COPY . .

# Éviter les fichiers inutiles dans l’image (optionnel mais propre)
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Lancer FastAPI
CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
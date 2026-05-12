# Guide de contribution

Merci de l'intérêt que vous portez au projet **Accidents Routiers — MLOps Pipeline** !  
Ce document décrit les étapes pour contribuer proprement.

---

## Prérequis

- Python 3.10+
- Git
- Docker Desktop 24+
- Make

---

## Mise en place de l'environnement local

```bash
# 1. Forker le dépôt sur GitHub, puis cloner votre fork
git clone https://github.com/<votre-username>/mlops-accidents.git
cd mlops-accidents

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows

# 3. Installer les dépendances (mode éditable)
make install
```

---

## Workflow de contribution

### 1. Créer une branche

Toujours travailler sur une branche dédiée, jamais directement sur `main`.

```bash
git checkout -b feat/ma-fonctionnalite
# ou
git checkout -b fix/nom-du-bug
```

Convention de nommage :

| Préfixe | Usage |
|---------|-------|
| `feat/` | Nouvelle fonctionnalité |
| `fix/` | Correction de bug |
| `refactor/` | Refactoring sans changement de comportement |
| `docs/` | Documentation uniquement |
| `test/` | Ajout ou correction de tests |
| `chore/` | Maintenance (dépendances, CI, …) |

### 2. Développer

- Respecter le style de code existant (PEP 8, vérifiable via `make lint`).
- Chaque nouveau module doit avoir des tests dans `tests/`.
- Ne pas committer de données brutes ou de modèles entraînés (`.gitignore` en place).

### 3. Lancer les tests

```bash
make test
```

La couverture minimale requise est **60 %**. La CI bloquera toute PR en dessous de ce seuil.

### 4. Vérifier le style

```bash
make lint
```

### 5. Ouvrir une Pull Request

1. Pousser votre branche : `git push origin feat/ma-fonctionnalite`
2. Ouvrir une PR sur GitHub contre la branche `main`.
3. Remplir le template de PR :
   - **Contexte** : pourquoi ce changement ?
   - **Changements** : ce qui a été modifié.
   - **Tests** : comment tester manuellement ?
4. Attendre la revue et adresser les commentaires.

---

## Standards de code

### Structure des fichiers

```
src/
├── api/
│   ├── main.py          ← app FastAPI + lifespan
│   ├── schemas.py       ← modèles Pydantic (un fichier par domaine)
│   ├── metrics.py       ← état partagé
│   └── routers/
│       ├── monitoring.py
│       └── inference.py
├── config/config.py     ← configuration centralisée
├── data/                ← scripts de préprocessing
└── models/              ← scripts d'entraînement
```

**Règle : un fichier par responsabilité.** Ne pas ajouter de routes directement dans `main.py`.

### Docstrings

Format NumPy pour les fonctions publiques :

```python
def predict(features: AccidentFeatures) -> PredictionResponse:
    """Prédit la gravité d'un accident.

    Parameters
    ----------
    features : AccidentFeatures
        Les 28 variables caractérisant l'accident.

    Returns
    -------
    PredictionResponse
        Prédiction binaire, label, probabilité et niveau de confiance.
    """
```

### Logging

Utiliser le logger nommé `"accidents_api"` — ne pas utiliser `print()`.

```python
logger = logging.getLogger("accidents_api")
logger.info("event=mon_event param=%s", valeur)
```

---

## Signaler un bug

Ouvrir une [issue GitHub](https://github.com/waneib22/mlops-accidents/issues) avec :

- La version Python et les dépendances (`pip freeze`).
- Les étapes pour reproduire.
- Le message d'erreur complet.
- Le comportement attendu vs observé.

---

## Contact

Pour toute question, ouvrir une [issue GitHub](https://github.com/waneib22/mlops-accidents/issues).

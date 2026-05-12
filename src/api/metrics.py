"""État partagé en mémoire entre les routers de l'API.

Toutes les variables de ce module sont importées par référence, ce qui permet
à monitoring.py et inference.py de partager le même objet sans passer par des
singletons ou des injections de dépendances.
"""
import threading

ml_model: dict = {}
startup_time: float = 0.0
stats: dict = {"total": 0, "prioritaire": 0, "non_prioritaire": 0}
stats_lock = threading.Lock()

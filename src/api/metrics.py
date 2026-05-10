import threading

# Shared in-memory state
ml_model: dict = {}
startup_time: float = 0.0
stats: dict = {"total": 0, "prioritaire": 0, "non_prioritaire": 0}
stats_lock = threading.Lock()

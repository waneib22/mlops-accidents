import threading

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNTER = Counter(
    "accidents_predictions_total",
    "Total number of predictions made",
    ["label", "confidence"],
)
PREDICTION_LATENCY = Histogram(
    "accidents_prediction_latency_seconds",
    "Time spent processing a prediction request",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_LOADED_GAUGE = Gauge(
    "accidents_model_loaded",
    "Whether the ML model is currently loaded (1=yes, 0=no)",
)
RETRAIN_COUNTER = Counter(
    "accidents_retrain_requests_total",
    "Total number of retraining requests triggered",
)

# Shared in-memory state
ml_model: dict = {}
startup_time: float = 0.0
stats: dict = {"total": 0, "prioritaire": 0, "non_prioritaire": 0}
stats_lock = threading.Lock()

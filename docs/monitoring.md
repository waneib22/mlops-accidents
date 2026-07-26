# Monitoring — Prometheus & Grafana

Documentation de la stack de supervision de l'API de prédiction des accidents.

- **Prometheus** : http://localhost:9090 — collecte et stocke les métriques
- **Grafana** : http://localhost:3000 (admin / admin) — visualise les métriques
- Dashboard : *« Accidents MLOps — Monitoring API »*

---

## 1. Architecture

```
┌────────────────┐   scrape /metrics    ┌──────────────┐   requêtes     ┌──────────┐
│  API FastAPI   │ ───(toutes les 15s)─▶ │  Prometheus  │ ◀────PromQL──── │ Grafana  │
│ accidents_api  │                      │  :9090        │                │  :3000   │
│   :8000        │                      │ (séries temp.)│                │ (dashb.) │
└────────────────┘                      └──────────────┘                └──────────┘
```

1. L'API expose ses métriques au format Prometheus sur **`GET /metrics`**.
2. **Prometheus** interroge (« scrape ») cet endpoint toutes les 15 secondes et stocke les valeurs dans sa base de séries temporelles.
3. **Grafana** interroge Prometheus en **PromQL** et affiche les courbes.

Le tout est orchestré par [`docker-compose.yml`](../docker-compose.yml). Le câblage est **provisionné automatiquement** (datasource + dashboard chargés au démarrage, aucun clic manuel) :

| Fichier | Rôle |
|---------|------|
| [`prometheus.yml`](../prometheus.yml) | Cible à scraper (`accidents_api:8000`), intervalle 15 s |
| [`grafana/provisioning/datasources/prometheus.yml`](../grafana/provisioning/datasources/prometheus.yml) | Branche Grafana sur Prometheus |
| [`grafana/provisioning/dashboards/dashboard.yml`](../grafana/provisioning/dashboards/dashboard.yml) | Charge automatiquement les dashboards du dossier |
| [`grafana/provisioning/dashboards/accidents_dashboard.json`](../grafana/provisioning/dashboards/accidents_dashboard.json) | Le dashboard lui-même (10 panels) |

---

## 2. Métriques exposées par l'API

Deux familles, toutes servies sur le **même** endpoint `/metrics`.

### a) Métriques HTTP génériques

Générées automatiquement par `prometheus-fastapi-instrumentator` ([`api/main_api.py`](../api/main_api.py)) :

| Métrique | Type | Labels | Description |
|----------|------|--------|-------------|
| `http_requests_total` | Counter | `handler`, `method`, `status` | Nombre cumulé de requêtes |
| `http_request_duration_seconds` | Histogram | `handler`, `method`, `le` | Latence des requêtes (buckets) |
| `http_requests_in_progress` | Gauge | `handler`, `method` | Requêtes en cours de traitement |

> Le label `status` est **groupé** : il vaut `"2xx"`, `"4xx"`, `"5xx"` (et non `200`/`404`).

### b) Métriques ML métier (custom)

Définies manuellement pour superviser le **modèle**, pas seulement le serveur web :

| Métrique | Type | Labels | Description |
|----------|------|--------|-------------|
| `model_predictions_total` | Counter | `predicted_class` | Nombre de prédictions par classe de gravité prédite |
| `model_inference_duration_seconds` | Histogram | `le` | Temps d'inférence du modèle (`predict` seul, hors I/O HTTP) |

C'est cette famille qui rend la supervision **« MLOps »** : elle permet de détecter un **dérive de distribution** (ex. soudain 100 % de prédictions sur une seule classe) ou une **dégradation de performance** (inférence qui ralentit).

> ⚠️ Ne pas confondre `GET /metrics` (format Prometheus, scrapé par Prometheus) et `GET /model/metrics` (JSON lisible : accuracy / precision / recall / f1 sur le jeu de test).

---

## 3. Les panels du dashboard

### Ligne 1 — Indicateurs clés (`stat`)

| Panel | PromQL | Lecture |
|-------|--------|---------|
| **Requêtes totales** | `sum(http_requests_total{job="accidents_api"})` | Volume cumulé depuis le démarrage |
| **Requêtes / seconde** | `sum(rate(http_requests_total[1m]))` | Charge instantanée (débit) |
| **Latence p50** | `histogram_quantile(0.50, …duration…_bucket…)` | Temps de réponse médian |
| **Latence p99** | `histogram_quantile(0.99, …)` | Pire cas pour 1 % des requêtes (détecte les pics) |

> `rate(...[1m])` = variation par seconde du compteur sur la dernière minute.
> `histogram_quantile(q, ...)` = reconstruit le quantile `q` à partir des buckets de l'histogramme.

### Ligne 2 — Vue par endpoint (`timeseries`)

| Panel | PromQL | Lecture |
|-------|--------|---------|
| **Volume par endpoint** | `sum by (handler) (rate(http_requests_total[1m]))` | Quels endpoints sont les plus sollicités |
| **Latence par endpoint (p95)** | `histogram_quantile(0.95, sum by (handler, le) (rate(…_bucket[1m])))` | Quel endpoint est lent |

### Ligne 3 — Santé & charge

| Panel | PromQL | Lecture |
|-------|--------|---------|
| **Taux d'erreurs HTTP** | `sum by (status) (rate(http_requests_total{status=~"4..\|5.."}[1m]))` | Apparition d'erreurs 4xx/5xx |
| **Requêtes en cours** | `http_requests_in_progress{job="accidents_api"}` | Saturation / requêtes bloquées |

### Ligne 4 — Métriques ML

| Panel | PromQL | Lecture |
|-------|--------|---------|
| **Prédictions par classe** | `sum by (predicted_class) (rate(model_predictions_total[5m]))` | Distribution des classes prédites (détection de dérive) |
| **Temps d'inférence (p95)** | `histogram_quantile(0.95, sum by (le) (rate(model_inference_duration_seconds_bucket[5m])))` | Performance pure du modèle |

---

## 4. Lancer la stack

```bash
cd Template_MLOps_accidents
docker compose up --build -d        # démarre API + Prometheus + Grafana (+ MLflow)

# Générer du trafic pour remplir les courbes
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @src/models/test_features.json
```

Puis ouvrir :
- **Grafana** → http://localhost:3000 → dashboard *Accidents MLOps — Monitoring API*
- **Prometheus** → http://localhost:9090 → *Status → Targets* (la cible `accidents_api` doit être **UP**)

---

## 5. Dépannage

| Symptôme | Cause probable | Solution |
|----------|----------------|----------|
| Tous les panels vides | Cible Prometheus **DOWN** | `9090 → Status → Targets`. Vérifier que l'API tourne et que `prometheus.yml` pointe sur `accidents_api:8000` |
| Panels ML vides | Aucune prédiction encore faite | Appeler `POST /predict` au moins une fois |
| « No data » sur la latence | Pas assez de trafic | Les `rate()` ont besoin de ≥ 2 points sur la fenêtre `[1m]` |
| Erreurs de scrape après MAJ deps | Version d'instrumentator différente | Épingler `prometheus-fastapi-instrumentator==8.0.1` dans `requirements.txt` |
| Dashboard absent dans Grafana | Provisioning non monté | Vérifier le volume `./grafana/provisioning` dans `docker-compose.yml` |

---

## 6. Pistes d'amélioration (Phase suivante)

- **Alerting** : règles Prometheus (`alert.rules.yml`) + Alertmanager (ex. p99 > 1 s, taux d'erreur > 5 %).
- **Gauge accuracy** : panel interrogeant périodiquement `/model/metrics` pour suivre la qualité du modèle dans le temps.
- **node-exporter / cAdvisor** : métriques CPU/RAM des conteneurs.
- **Persistance Prometheus** : volume dédié pour conserver l'historique entre redémarrages.
</content>
</invoke>

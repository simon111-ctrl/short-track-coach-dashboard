# Short-Track Multi-Distance Coach Workspace

Streamlit web app for coach-facing post-race analysis of short-track speed skating races across 500m, 1000m, and 1500m.

The app consumes the exported model handoff package only. It does not retrain models or change model logic.

## Features

- Chinese / English interface switch
- Distance switch for 500m, 1000m, and 1500m
- Manual single-race input
- CSV / Excel batch analysis
- Automatic feature engineering per distance
- Automatic model routing for grade, advancement, max round, final entry, rhythm cluster, tactical style, key lap, and anomaly risk
- Coach-facing results, explanations, notes, and downloadable reports
- Visible model version, training rows, and applicable distance

## Model Package

The deployed model assets live in `model_package/`:

- `model_manifest.json`
- `models/<distance>_<task>/model.joblib`
- `models/<distance>_<task>/features.json`
- `src/feature_engineering.py`
- `reports/model_metrics.csv`
- `reports/model_explanations.csv`
- `reports/cluster_centers_<distance>_style_cluster.csv`
- `examples/`

Large `model.joblib` files are tracked with Git LFS.

## Local Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Validation

```powershell
python -m pytest tests/test_short_track_service.py -q
python -m py_compile app.py short_track_service.py
```

The service test loads the model package, engineers features, and verifies coach outputs from the exported sample input.

## Streamlit Cloud

Deploy this repository with:

- Entry point: `app.py`
- Python dependencies: `requirements.txt`
- Model assets: included via Git LFS

After deployment, verify:

1. Page loads publicly.
2. Model version is visible in the sidebar.
3. 500m, 1000m, and 1500m sample inputs produce results.
4. CSV template download works.
5. Batch upload returns an Excel analysis file.

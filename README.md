# Short-Track Elite Feature Predictor

Streamlit web app for coach-facing short-track speed skating analysis across 500m, 1000m, and 1500m.

This version uses the retrained gender-specific model package. The web app does not retrain models and does not use the old mixed-gender unified model.

## Features

- Gender selection: male / female
- Distance selection: 500m, 1000m, 1500m
- Separate model routing by gender + distance + task
- Manual single-race input
- CSV / Excel batch analysis
- Automatic total-time calculation from lap or segment times
- Time input formats: `mm:ss:SSS`, `mm:ss`, `m:ss`, `ss`, `ss.sss`, or pure seconds such as `42.318`
- Feature alignment checks against each selected model's `features.json`
- Coach-facing result, explanation, advice, chart, and downloadable report

## Model Package

The deployed assets live in `model_package/`:

- `web_model_manifest.json`
- `model_manifest.json`
- `models/male/<distance>/<task>/model.joblib`
- `models/male/<distance>/<task>/features.json`
- `models/female/<distance>/<task>/model.joblib`
- `models/female/<distance>/<task>/features.json`
- `src/feature_engineering.py`
- `reports/model_explanations.csv`
- `metrics/gender_distance_model_comparison.csv`
- `examples/example_input_<gender>_<distance>_<task>.csv`

Large `model.joblib` files are tracked with Git LFS.

## Local Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

If dependencies are missing in Anaconda:

```powershell
conda install -c conda-forge streamlit pandas numpy scikit-learn joblib plotly openpyxl pytest
```

or:

```powershell
python -m pip install -r requirements.txt
```

## Validation

```powershell
python -m pytest tests/test_short_track_service.py -q
python -m py_compile app.py short_track_service.py
```

The service tests verify time parsing, gender-specific model loading, feature construction, prediction output, and all 6 gender-distance combinations.

## Streamlit Cloud

Deploy this repository with:

- Entry point: `app.py`
- Python dependencies: `requirements.txt`
- Model assets: the full `model_package/` directory, including `models/male`, `models/female`, `reports`, `metrics`, `examples`, `src`, and manifests
- Git LFS enabled for `model_package/models/**/model.joblib`

After deployment, verify male and female predictions separately for 500m, 1000m, and 1500m.

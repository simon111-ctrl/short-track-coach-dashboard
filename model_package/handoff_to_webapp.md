# Handoff To Webapp

This package contains the retrained gender-specific short-track models for 500m, 1000m, and 1500m.

The app must not use the old mixed-gender model package. It must select models by:

1. `gender`: `male` or `female`
2. `distance`: `500m`, `1000m`, or `1500m`
3. `task`: `advancement`, `grade`, `max_round`, `final_entry`, `style_cluster`, `tactical_style`, `key_lap`, or `risk_detection`

Model folders use:

```text
models/<gender>/<distance>/<task>/model.joblib
models/<gender>/<distance>/<task>/features.json
```

Use `web_model_manifest.json` or `model_manifest.json` as the entry point. The manifest contains `gender`, `distance`, `task`, `n_training_rows`, `feature_columns`, and report paths for each model.

Important constraints:

- Gender is a model selector, not a regular model feature.
- Do not mix male and female models or reference statistics.
- Do not send athlete name, round, qual code, source URL, or gender into the model feature matrix unless a selected `features.json` explicitly requires it.
- Feature order must exactly follow the selected `features.json`.
- Grade models intentionally exclude total-time leakage features.
- `style_cluster` and `risk_detection` are unsupervised models and do not have accuracy/F1 metrics.
- Prediction advice should state that the result is compared only with the selected gender's reference group.

Files required for Streamlit deployment:

- `app.py`
- `short_track_service.py`
- `requirements.txt`
- `model_package/web_model_manifest.json`
- `model_package/model_manifest.json`
- `model_package/models/male/**`
- `model_package/models/female/**`
- `model_package/src/feature_engineering.py`
- `model_package/feature_lists_by_distance.json`
- `model_package/reports/model_explanations.csv`
- `model_package/metrics/gender_distance_model_comparison.csv`
- `model_package/examples/example_input_<gender>_<distance>_advancement.csv`

Large `model.joblib` files should remain tracked with Git LFS.

# Handoff To Webapp Thread

This package contains newly trained 10-year short-track models for 500m, 1000m, and 1500m. It includes the previous advancement and grade models plus the newly requested max-round, final-entry, rhythm-cluster, tactical-style, key-lap, and risk-detection models. It is for model consumption only; no web deployment work is included.

Use `model_manifest.json` as the entry point. Each model folder contains `model.joblib` and `features.json`.

Webapp integration steps:
1. Load the proper distance/task model from `models/<distance>_<task>/model.joblib`.
2. Recreate features with `src/feature_engineering.py` or mirror the same formulas.
3. Select the exact feature order from `features.json`.
4. For classifiers, return predicted label and class probabilities.
5. For `style_cluster`, return the cluster ID and optionally use the cluster center table in `reports/`.
6. For `risk_detection`, return `risk_score` from `-decision_function(X)` and `risk_label` from `predict(X) == -1`.

Model tasks included per distance:
- `advancement`: official Qual-based advancement probability.
- `grade`: quartile-based performance grade with leakage-controlled features.
- `max_round`: predicted highest round score reached in the same event.
- `final_entry`: probability of reaching Ranking Finals / Final B / Finals / Final A.
- `style_cluster`: unsupervised rhythm and pacing cluster.
- `tactical_style`: rule-trained tactical style class.
- `key_lap`: rule-trained key lap class.
- `risk_detection`: unsupervised anomaly/risk model.

Important constraints:
- Do not send athlete name or athlete ID into the model. They are metadata only.
- Do not send `qual` into the model. It is label source only.
- Grade models intentionally avoid total-time leakage features.
- Advancement models are post-race/reference models using full lap information; they are not early-race live predictors.
- `tactical_style` and `key_lap` use rule-generated labels, not official labels.
- `style_cluster` and `risk_detection` are unsupervised models and do not have accuracy/F1 metrics.

Files the webapp thread needs:
- `model_manifest.json`
- `models/`
- `src/feature_engineering.py`
- `feature_lists_by_distance.json`
- `input_fields.md`
- `label_rules.md`
- `reports/model_metrics.csv`
- `reports/model_explanations.csv`
- `reports/cluster_centers_<distance>_style_cluster.csv`
- `examples/`

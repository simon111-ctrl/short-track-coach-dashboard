# 500 m World Cup Short-Track Coach Dashboard

This Streamlit app deploys the final CatBoost models from the short-track speed skating study as a public coach-facing post-race decision-support tool.

## Models

- Performance grade model: CatBoost, athlete-grouped weighted F1 = 0.920, macro AUC = 0.982.
- Advancement reference model: CatBoost, ADV excluded, athlete-grouped weighted F1 = 0.823, AUC = 0.894.

## App Inputs

Coaches enter official total time, five lap times, and five lap positions for a 500 m race. The app automatically computes lap-to-lap speed changes, position changes, variability, start and sprint ratios, and position improvement.

## Local Run

```powershell
streamlit run app.py
```

## Deployment

Deploy this folder to Streamlit Community Cloud with `app.py` as the entrypoint.

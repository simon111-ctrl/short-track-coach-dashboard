# Input Fields

The web app uses gender-specific short-track models. Required user-facing inputs are:

- `gender`: required model selector. Allowed values are `male` and `female`; the UI also accepts 男 / 女 / Male / Female / Men / Women.
- `distance`: required model selector. Allowed values are `500m`, `1000m`, and `1500m`.
- `official_total_time`: optional. If blank, the app uses the sum of lap times.
- `lapN_time`: required for every lap in the selected distance. The app accepts `mm:ss:SSS`, `mm:ss`, `m:ss`, `ss`, `ss.sss`, or pure seconds.
- `lapN_position`: required for every lap in the selected distance.
- `athlete_name`, `round`, `qual_code`: optional metadata for display and audit. These are not sent as model features.

Distance-specific lap counts:

- `500m`: laps 1-5
- `1000m`: laps 1-9
- `1500m`: laps 1-14

The model input DataFrame is built after feature engineering. Exact feature names and order come from each selected `models/<gender>/<distance>/<task>/features.json`.

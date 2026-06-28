# Input Fields

Required raw fields:
- 姓名: athlete name, used only for grouped validation and audit.
- 总成绩: official total time. Can be blank when all lap times are present; then it is reconstructed from lap sums.
- qual: official qualification code used to define advancement labels.
- 轮次: race round, used for excluding Final A/B from advancement training and for audit.
- 第N圈成绩 and 第N圈位置: all lap time and lap position fields are required for the distance.
- 赛季, 比赛名称, 比赛地点, 比赛日期, 项目, 国家/地区, source_url: metadata for audit, time validation, and traceability.

Distance-specific lap counts:
- 500m: laps 1-5.
- 1000m: laps 1-9.
- 1500m: laps 1-14.

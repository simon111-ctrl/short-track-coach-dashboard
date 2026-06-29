# Label Rules

All labels were built separately within each gender and distance group. A male prediction uses only male-trained models and male reference statistics; a female prediction uses only female-trained models and female reference statistics.

## Advancement label

Positive labels are defined from the official `qual` field.

Positive codes:
`Q`, `QA`, `QB`, `QAA`, `QAB`, `QBA`, `QBB`, `q`, `qA`, `qB`, `ADV`, `ADA`, `ADVA`, `ADVB`.

Final rounds are excluded from advancement training because they do not represent qualification into a next round.

## Grade label

Within each gender and distance, `reconstructed_total_time` is divided into quartiles. Label `3` is the fastest/top quartile and label `0` is the slowest/bottom quartile.

To control label leakage, grade models do not use official total time, reconstructed total time, mean lap time, or raw absolute lap times as predictors. They use pacing-shape, lap-delta, position, and distance-specific tactical features.

## Max round label

For each athlete within the same gender, distance, and event URL, the label is the highest normalized round reached by that athlete in that event.

Round score map:

- 1: Preliminaries
- 2: Rep. Heats
- 3: Heats
- 4: Rep. Quarterfinals
- 5: Quarterfinals
- 6: Rep. Semifinals
- 7: Semifinals
- 8: Ranking Finals / Final B
- 9: Finals / Final A

Athlete name is used only to construct the training label and grouped validation. It is not a model input.

## Final entry label

Positive label is assigned when `max_round_score >= 8`, meaning the athlete reached Ranking Finals, Final B, Finals, or Final A in that event.

## Rhythm style cluster

This is an unsupervised KMeans model trained within the selected gender and distance group. Cluster IDs describe pacing, rhythm, and position-control patterns.

## Tactical style label

This is a rule-generated training label derived from position-control features:

- `front_runner`
- `late_attacker`
- `stable_controller`
- `volatile_risk`
- `chaser`

## Key lap label

This is a rule-generated lap number. The selected lap has the largest combined signal from lap-time deviation, position change, and final-lap transition.

## Risk detection

This is an unsupervised IsolationForest model. It outputs a risk/anomaly score from lap variability, position instability, late-race deterioration, and total-time consistency features.

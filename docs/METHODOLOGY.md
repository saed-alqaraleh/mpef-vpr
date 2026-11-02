# Monotone Probabilistic Evidence Fusion

The final frozen contract contains 21 features: four global, five local, nine geometric/spatial and three cross-evidence agreement features. Every feature has a predeclared direction.

Each feature is oriented so increasing values support relevance; an isotonic likelihood map is fitted on FIT cities, its log-odds output is centered by its FIT-weighted mean, and nonnegative coefficients combine the evidence in a logistic fusion head. VALIDATION selects regularization, while the independent CALIBRATION split learns a final isotonic probability map. The source fit used 14 FIT, five VALIDATION and four CALIBRATION GSV-Cities. Selected L1=0 and L2=0.001 for both DISK and ALIKED backends.

Ranking is evaluated using the original uncalibrated decision score, while the separate calibration map is used for proper probability metrics. Calibration-induced ties must never alter ranked candidates.

Primary K=100 and sensitivity K=50 require separate raw-list 21-feature reconstruction. No K=50 feature table should be obtained by truncating K=100-derived feature rows. Physical queries, not retriever/backend replications, are the resampling unit. No synthetic verification example is a benchmark result.

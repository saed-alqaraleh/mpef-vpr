# Methodology

## Overview

Monotone Probabilistic Evidence Fusion (MPEF) is a fixed-candidate post-matching model for visual place recognition. Given a global-retrieval shortlist and its local geometric matching evidence, MPEF constructs a 21-dimensional description of each query–candidate pair, then learns direction-constrained relevance evidence.

## Evidence construction

The feature contract is recorded in [`configs/feature_contract.json`](../configs/feature_contract.json) and implemented in `src/mpef_vpr/features.py`.

| Evidence group | Count | Examples |
| --- | ---: | --- |
| Global retrieval | 4 | Similarity/rank percentiles, shortlist margin and entropy |
| Local matching | 5 | Inlier count and ratio, match confidence, robust-estimation score |
| Spatial geometry | 9 | Hull coverage, inlier spatial entropy, reprojection residuals, degeneracy |
| Rank agreement | 3 | Joint support, disagreement, list-level Kendall concordance |
| **Total** | **21** | |

Each feature has a declared favorable direction. Geometric observations are computed from the local matcher inlier measurements; global list statistics and agreement features depend on the entire candidate shortlist.

## Monotone evidence fusion

For each feature, a direction-oriented isotonic map estimates a relevance likelihood on the FIT split. Log-odds evidence is mean-centered and combined with a nonnegative additive logistic head. Nonnegative weights preserve the specified direction of each learned feature contribution.

Regularization is selected using the geographically disjoint VALIDATION split. The selected backend models use `l1 = 0` and `l2 = 0.001` under the recorded validation criterion.

The corresponding probability maps are fitted on a separate CALIBRATION split using monotone isotonic regression. The development protocol separates 14 FIT, 5 VALIDATION and 4 CALIBRATION GSV-Cities. Both calibration files are linked by SHA-256 to their corresponding frozen ranking model.

## Ranking and probability estimation

Candidate ranking follows the frozen **uncalibrated model decision score**, with deterministic secondary ordering by original global rank and candidate ID. Held-out isotonic maps are used **only** for match-probability reporting and proper scoring. This separation prevents calibration plateaus from changing ranking.

## Candidate-list depth

The primary shortlist is `K=100`, with `K=50` evaluated as a sensitivity setting. The global percentile, list entropy, local percentile and agreement evidence depend on shortlist composition. Accordingly, each K value requires a separately constructed feature matrix.

## Evaluation conventions

The original protected test suite covers MSLS, Tokyo 24/7, Gardens Point, AmsterTime, SPEDTest, and 17-Places. Candidate log loss and Recall@1 are distinct primary endpoints. Reported inference uses physical queries as statistical resampling units, grouping observations of the same query across retriever and matcher configurations.

Further details are given in [Results](RESULTS.md) and [Reproducibility](REPRODUCIBILITY.md).

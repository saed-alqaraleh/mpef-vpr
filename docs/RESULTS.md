# Results and interpretation

## Archived original six-dataset comparisons

The source audit reports equal-dataset K=100 candidate log loss of **0.10956** for MPEF and **0.15741** for calibrated logistic stacking under the documented shared 1e-7 clipping convention. The full per-candidate evaluation arrays are **not** present in this staging repository; these are preserved archival summaries, **not independently rerun six-dataset numbers**.

The saved \`results/original_evaluation/final_co_primary_inference_v1.csv\` contains the archived inferential summary. The H1 MPEF-minus-shallow-pairwise-family Recall@1 contrast is **negative**, although H2 candidate-log-loss contrast favors MPEF. Claims should distinguish superior proper-score evidence from the strongest competing ranking head. Results for the larger capacity models should not be portrayed as evidence of MPEF universally outperforming every probabilistic comparator.

## Tokyo 24/7 modern retrieval only

| Retriever | Queries | Gallery | Recall@1 | Recall@5 | Recall@10 | mAP@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SALAD | 147 | 16,140 | 92.52% | 97.96% | 99.32% | 0.73358 |
| BoQ | 147 | 16,140 | 95.24% | 100.00% | 100.00% | 0.80481 |

These figures are independently averaged from saved per-query metric records by \`scripts/verify_release.py\`, under the frozen SUTD 147/16,140 and ok/junk protocol. **They are not MPEF post-matching fusion results.** The October 31 missing-Tokyo count in the earlier handoff was superseded by the November 2 Tokyo-completion record; the updated result is included here.

## MSLS 25-m localization

The preserved \`results/original_evaluation/MSLS_MPEF_VS_ORIGINAL_POSE_PAIRED.csv\` contains selected paired confidence intervals from 740 physical queries. Raw pose predictions and geometric localization evidence are not included.

## Transferred SALAD–ALIKED fusion

A separate three-domain zero-refit transfer was described in a subsequent handoff. Its full source predictions and raw matching evidence are not included here, and this staging repository therefore makes no claim of having replayed that transfer.

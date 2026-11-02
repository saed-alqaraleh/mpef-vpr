# Evaluation results

## Matching-aware evidence fusion

MPEF was evaluated under the original fixed-candidate matching-aware VPR protocol across Pittsburgh30k, MSLS, Tokyo 24/7, Oxford RobotCar, Gardens Point, and VPAir. The [archived co-primary inference table](../results/original_evaluation/final_co_primary_inference_v1.csv) preserves the paired statistical contrasts and confidence intervals.

The archived pooled candidate log-loss summary under equal-dataset weighting reports **0.10956** for MPEF and **0.15741** for the calibrated logistic-stacking comparator. This comparison assesses match-probability quality and is distinct from ranking performance.

For ranking, the recorded MPEF-minus-shallow-pairwise-ranker Recall@1 contrast was **−0.00901** (95% confidence interval **[−0.01452, −0.00361]**), indicating that the shallow pairwise comparator ranked more accurately on that endpoint. In the calibrated probability comparison, the recorded logistic-minus-MPEF candidate log-loss contrast was **+0.06598** (95% confidence interval **[0.05798, 0.07433]**), favoring MPEF on that specific proper-score comparison.

These contrasts concern **different objectives and comparator families**; neither result substitutes for the other. For the capacity-expanded logistic comparator, a subsequent archived comparison found a smaller log-loss advantage over MPEF. All reported comparisons retain their specific model capacities and evaluation protocols.

The repository includes original statistical summary CSV files but does not include the complete six-domain raw candidate prediction tables required to rerun those archived model comparisons from scratch.

## Tokyo 24/7: modern retrieval reference

Modern retrieval-only experiments were recorded under the corrected 147-query, 16,140-gallery SUTD protocol with the declared junk handling.

| Method | Query count | Recall@1 | Recall@5 | Recall@10 | mAP@100 |
| :-- | --: | --: | --: | --: | --: |
| SALAD | 147 | 92.52% | 97.96% | 99.32% | 0.73358 |
| BoQ | 147 | 95.24% | 100.00% | 100.00% | 0.80481 |

These values are independently averaged from the [SALAD](../results/modern_retrieval/tokyo_salad_metrics_by_query.csv) and [BoQ](../results/modern_retrieval/tokyo_boq_metrics_by_query.csv) query-level results using `scripts/verify_release.py`.

**SALAD and BoQ are evaluated here as retrieval-only upstream references.** The table does not represent local geometric verification or MPEF fusion.

## MSLS paired localization

The [MSLS paired 25-m analysis](../results/original_evaluation/MSLS_MPEF_VS_ORIGINAL_POSE_PAIRED.csv) records backend/retriever-specific paired localization summaries on 740 physical queries per condition. These summary metrics are stored separately from the full raw pose outputs.

## Interpretation

The archived results describe calibrated candidate matching, candidate ranking, retrieval-only reference quality and geometric localization under their respective protocols. The files and scripts preserve these distinctions rather than pooling quantities with different definitions.

# MPEF — Monotone Probabilistic Evidence Fusion

### Matching-Aware Visual Place Recognition

[![Continuous integration](https://github.com/saed-alqaraleh/mpef-vpr/actions/workflows/tests.yml/badge.svg)](https://github.com/saed-alqaraleh/mpef-vpr/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Method](https://img.shields.io/badge/features-21-596b79)

**MPEF** combines global retrieval, local geometric verification, spatial consistency, and rank agreement to produce interpretable candidate rankings and calibrated match probabilities for visual place recognition (VPR).

The implementation follows a fixed-candidate post-matching pipeline. It learns monotone evidence transformations, combines them through nonnegative logistic fusion, and calibrates the resulting probabilities on geographically disjoint data. Candidate ranking uses the underlying fusion score, while probability assessment uses the independent calibration map.

## Highlights

- **21 directional evidence features:** global retrieval (4), local matching (5), spatial geometry (9), and agreement (3).
- **Monotone probabilistic fusion:** feature-wise isotonic maps and nonnegative fusion coefficients.
- **Independent probability calibration:** held-out isotonic maps, separate from the ranking score.
- **Two local-feature backends:** DISK–LightGlue–MAGSAC++ and ALIKED–LightGlue–MAGSAC++.
- **Frozen experimental artifacts:** fitted model records, calibration maps, input schemas, and numerical integrity checks.
- **Reproducible evaluation components:** automated tests, dataset-level summaries, and query-level retrieval metrics.

## Method

\`\`\`text
Query image
    │
    ├── Global image retrieval ──► Fixed candidate shortlist (K = 100)
    │                                      │
    └──────────────────────────────► Local geometric matching
                                           │
                                Retrieval and matching evidence
                                           │
                             21 directional evidence features
                                           │
                            Feature-wise monotone likelihood maps
                                           │
                             Nonnegative probabilistic fusion
                                           │
                              ┌────────────┴─────────────┐
                              ▼                          ▼
                       Candidate ranking        Held-out calibration
                       (fusion score)           (match probabilities)
\`\`\`

The primary evaluation uses **K = 100** candidates. The K = 50 sensitivity analysis requires a separately constructed 50-candidate evidence representation; candidate-list-dependent features cannot be obtained by truncating the K = 100 feature matrix.

## Installation

Python 3.10 or later is required.

\`\`\`bash
git clone https://github.com/saed-alqaraleh/mpef-vpr.git
cd mpef-vpr
python -m pip install -e ".[test]"
\`\`\`

## Quick start

Run a deterministic software example through the frozen model and calibration maps:

\`\`\`bash
python scripts/quickstart.py --backend disk
python scripts/quickstart.py --backend aliked
\`\`\`

The quick-start example uses synthetic matching evidence to demonstrate the API. Research metrics are reported separately from this software test.

To score a real fixed-candidate evidence archive:

\`\`\`bash
python scripts/score_evidence.py \
  --backend disk \
  --k 100 \
  --evidence path/to/evidence_k100.npz \
  --output results/scored_candidates.csv
\`\`\`

The input archive must satisfy [the evidence schema](schemas/evidence_npz.schema.json). For each query, the output contains the deterministic candidate order and corresponding held-out calibrated relevance probabilities.

## Verification

\`\`\`bash
python -m pytest -q tests
python scripts/verify_release.py
python scripts/check_firewall.py
\`\`\`

The verification suite covers feature construction, monotonicity, frozen-parameter integrity, ranking conventions, synthetic end-to-end scoring, and numerical aggregation of the archived Tokyo 24/7 retrieval results. The SHA-256 digests of the two fitted models and both independent calibration maps are checked against the corresponding experimental records.

## Results

### Tokyo 24/7: retrieval-only reference

The archived modern-retriever evaluation uses **147 queries**, **16,140 gallery images**, and the specified SUTD relevance/junk handling.

| Retriever | Recall@1 | Recall@5 | Recall@10 | mAP@100 |
| :-- | --: | --: | --: | --: |
| SALAD | 92.52% | 97.96% | 99.32% | 0.73358 |
| BoQ | 95.24% | 100.00% | 100.00% | 0.80481 |

These are **retrieval-only reference results**, not post-matching MPEF outputs. The [query-level tables](results/modern_retrieval/) and numerical verification script support their aggregation. Original six-domain evidence-fusion summaries and the MSLS paired-pose comparison are documented in [Results](docs/RESULTS.md).

## Repository contents

| Path | Contents |
| :-- | :-- |
| \`src/mpef_vpr/\` | Features, monotone fusion, ranking, calibration, evidence I/O |
| \`models/\` | Frozen DISK/ALIKED models and held-out calibration maps |
| \`scripts/\` | Examples, evidence scoring, integrity and numerical checks |
| \`configs/\`, \`schemas/\` | Feature definitions and machine-readable input/output contracts |
| \`results/\` | Recorded statistical summaries and Tokyo query-level metrics |
| \`tests/\` | Unit, integration, and frozen-artifact regression tests |
| \`docs/\` | Method, evaluation, datasets, and reproducibility documentation |

## Research documentation

- [Methodology](docs/METHODOLOGY.md)
- [Evaluation results](docs/RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Data and dependencies](docs/DATA_SOURCES.md)

The repository provides the MPEF evidence-fusion implementation and archived results. Full image-to-evidence benchmark reconstruction additionally uses separately obtained benchmark images, retrieval checkpoints, and local matching dependencies, as described in the reproducibility documentation.

## Citation and licensing

Use the versioned archival DOI associated with the relevant GitHub release once registered. Reuse permissions for the source and fitted artifacts are governed by the repository [LICENSE](LICENSE); upstream datasets and neural-network checkpoints retain their own terms.

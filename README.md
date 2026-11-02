# MPEF: Monotone Probabilistic Evidence Fusion

**Research implementation and frozen-scoring artifacts for matching-aware visual place recognition.**

> **Private internal release candidate — not publicly licensed.** The recovered implementation carries an all-rights-reserved rights notice. Source/model redistribution and any public release require explicit rights-holder approval. Full six-benchmark end-to-end reconstruction is not established by the included files.

Visual place recognition produces fixed retrieval lists and local geometric-verification evidence that are not inherently calibrated relevance probabilities. MPEF uses 21 oriented retrieval, matching, spatial, and agreement features with feature-wise isotonic likelihood maps, nonnegative logistic fusion, and held-out isotonic probability calibration.

**Pipeline:** Global retrieval → local matching → 21-feature evidence contract → monotone likelihood maps → nonnegative fusion → (a) candidate ranking; (b) independently calibrated relevance probabilities.

## Verified scope
- The two selected frozen backend models and two held-out calibration files are preserved and checked against original SHA-256 freeze digests.
- A deterministic 21-feature builder, typed raw-evidence loader, frozen scoring API, and software regression tests are included.
- Retrieval-only Tokyo 24/7 SALAD and BoQ metrics are recomputed from 147 saved query-level rows per method under the 16,140-gallery protocol.
- The original complete six-dataset matching evidence, third-party matcher checkpoints, and all underlying images are **not** included. Retrieval-only output is not an MPEF fusion result.

## Installation and validation

    python -m pip install -e '.[test]'
    python -m pytest -q tests
    python scripts/verify_release.py
    python scripts/quickstart.py --backend disk

The quick start is a **synthetic-data software smoke test only**, not a reported experiment. For independently generated raw matching evidence:

    python scripts/score_evidence.py --backend aliked --k 100 --evidence path/to/raw_k100_evidence.npz --output scored.csv

For K=50, reconstruct raw-candidate features from a genuine K=50 shortlist; do not truncate a K=100 feature matrix.

## Result scope

| Task | Frozen protocol | Report |
| --- | --- | --- |
| Artifact integrity | DISK and ALIKED models/calibrators | 4/4 SHA-256 match |
| Tokyo SALAD retrieval-only | 147 queries, 16,140 gallery | Recall@1 = 92.52% |
| Tokyo BoQ retrieval-only | 147 queries, 16,140 gallery | Recall@1 = 95.24% |
| MPEF original six-domain suite | K=100 four pairing design | Archived summaries provided; full replay not included |
| MSLS geographic localization | Paired 25-m analysis | Archived summary only |

Read [methodology](docs/METHODOLOGY.md), [scope](docs/REPRODUCIBILITY.md), [results](docs/RESULTS.md), [upstream sources](docs/DATA_SOURCES.md) and [release gates](docs/RELEASE_GATES.md).

## Rights and citation

Source and model files are in this **private** repository for internal research review. Neither these original files nor third-party datasets/weights should be publicly redistributed without appropriate rights. No public license, DOI, publication status, or fully independent end-to-end reproduction is asserted. A final citation will be added after authorship/manuscript metadata are confirmed.

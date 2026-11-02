# Reproducibility

## Software environment

- Python 3.10+
- NumPy, SciPy, scikit-learn
- pytest for automated validation

\`\`\`bash
python -m pip install -e ".[test]"
python -m pytest -q tests
python scripts/quickstart.py --backend disk
python scripts/verify_release.py
\`\`\`

Continuous integration runs the package tests, frozen-artifact verification, and a deterministic scoring demonstration.

## Frozen model files

The \`models/\` directory contains the fitted DISK and ALIKED fusion parameters and their held-out isotonic calibration maps. The verification script checks the SHA-256 values against the recorded final-model and calibration freezes. A loader refuses a modified model or calibration file.

## Evidence-to-predictions workflow

1. Retrieve a fixed \`K=100\` shortlist using the selected global image retriever.
2. Run local geometric verification with the specified local feature/matching backend.
3. Export a versioned NPZ archive satisfying \`schemas/evidence_npz.schema.json\`. Arrays describe global similarities and ranks, inlier/match counts, confidence, spatial inliers, residuals and geometry validity.
4. Extract the full 21-feature matrix using \`mpef_vpr.features.build_feature_matrix\`.
5. Apply the corresponding frozen model. Rank using the underlying decision score and report probabilities using the held-out calibration map.
6. Evaluate with the frozen dataset-specific relevance protocol, separating candidate ranking endpoints from probabilistic endpoints.

\`\`\`bash
python scripts/score_evidence.py \
  --backend aliked --k 100 \
  --evidence path/to/your_evidence.npz \
  --output predictions.csv
\`\`\`

For K=50, extract the features from the K=50 raw shortlist rather than slicing features from K=100.

## Available numerical records

The repository preserves original evaluation summaries and the Tokyo 24/7 modern retrieval-only per-query metric records. The latter are independently averaged into R@1/5/10 and mAP@100 by \`scripts/verify_release.py\`. This is **aggregation replay of saved observations**, not a fresh extraction of image descriptors.

Original six-dataset image-to-match evaluation also requires the benchmark images, original retriever and matcher checkpoints, dataset protocol manifests and candidate-level matching evidence. These inputs are maintained by their original dataset/project sources and are not embedded in this compact code repository. Accordingly, saved summaries should be distinguished from metrics independently recomputed by this repository's scripts.

## Dataset and statistical controls

GSV-Cities fit, validation, and calibration cities are disjoint. Final selection and held-out calibration are frozen prior to protected-set scoring. Ranking uses raw fusion scores; probability metrics use calibration. Results for the two local backends and the various global retrieval configurations retain their documented experimental identities.

## Integrity

The source/model/calibration chain is identified by file hashes, explicit feature ordering, typed inputs and deterministic ranking. The test suite checks monotone orientations and the agreement between model objects and saved calibration maps.

# Reproducibility levels and scientific boundaries

**Tier 0 — frozen-model/software verification (automated).** Run \`python -m pytest -q tests\`, \`python scripts/verify_release.py\`, and \`python scripts/quickstart.py --backend disk\`. Saved model/calibrator bytes are checked against the original validation-selected source and calibration freeze hashes. The quick start consumes **synthetic input solely for smoke testing**.

**Tier 1 — archived retrieval-only numerical replay (automated).** \`scripts/verify_release.py\` recomputes R@1/5/10 and all-positives-normalized mAP@100 averages using 147 saved Tokyo 24/7 per-query rows for SALAD and for BoQ; it compares those aggregates to frozen saved summaries. This does not independently regenerate image descriptors or validate how the CSV files were originally constructed.

**Tier 2 — scoring authentic raw evidence (provided interface).** Create a schema-compliant \`.npz\` containing the original raw retriever/matcher evidence and invoke \`python scripts/score_evidence.py --backend disk --evidence /path/to/raw_k100.npz --k 100 --output scored.csv\`. Ranking uses uncalibrated MPEF scores, while held-out isotonic calibration supplies candidate probabilities. A correct K=50 run separately extracts all 21 list-dependent features from the K=50 raw shortlist.

**Tier 3 — complete original experiment recreation (not demonstrated here).** Reproducing all six protected datasets requires licensed images, frozen relevance and junk protocols, CosPlace/R2Former network models, DISK/ALIKED+LightGlue+MAGSAC++ raw matching, evidence generation and the original execution records. This repository does not supply complete end-to-end image-to-pose or image-to-ranking inference orchestration, nor the per-candidate original six-domain evidence necessary for numerical replication.

**Statistical safeguards.** Four original retriever–matcher configurations, SALAD/BoQ retrieval-only scores, zero-refit SALAD–ALIKED transfer and MSLS 25-m localization are distinct experimental tasks. Physical queries are the inferential unit. The original generic monotone comparator and higher-capacity capacity extensions are not interchangeable.

**Rights gate.** The recovered source was marked all rights reserved. Repository access is private and does not provide a public license. Any public release requires rights-holder authorization for source, fitted models and distributable evaluation artifacts, followed by manuscript consistency review.

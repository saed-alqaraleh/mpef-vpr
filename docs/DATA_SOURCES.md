# Datasets and upstream dependencies

MPEF operates on fixed global-retrieval shortlists and the corresponding local matching evidence. Benchmark images and third-party neural checkpoints are obtained from their official distributors under their respective terms.

## Development data and retrievers

- [GSV-Cities](https://github.com/amaralibey/gsv-cities): geography-separated fit, validation and calibration roles.
- [CosPlace](https://github.com/gmberton/CosPlace): global visual place retrieval.
- [R2Former](https://github.com/yanyan-bit/R2Former): global place representation, subject to its published implementation requirements.

## Local matching

- [DISK](https://github.com/cvlab-epfl/disk)
- [ALIKED](https://github.com/Shiaoming/ALIKED)
- [LightGlue](https://github.com/cvg/LightGlue)
- [Hierarchical Localization (hloc)](https://github.com/cvg/Hierarchical-Localization)

The protocol uses a deterministic local verification configuration with the declared keypoint cap, image resizing, matching confidence threshold and MAGSAC++ settings. Frozen MPEF fusion parameters are stored under `models/` and are separate from these upstream networks.

## Retrieval-only context

- [SALAD](https://github.com/serizba/salad)
- [Bag of Queries (BoQ)](https://github.com/amaralibey/Bag-of-Queries)

The saved Tokyo 24/7 retrieval-only evaluation has 147 physical queries and 16,140 reference images. The [result manifest](../results/modern_retrieval/FINAL_TOKYO_COMPLETION.json) records the image-manifest checksums, model checkpoint fingerprints, and evaluation scope.

## Protected evaluation collections

MSLS, Tokyo 24/7, Gardens Point, AmsterTime, SPEDTest, and 17-Places are evaluated under their frozen original dataset protocols. Protocol-specific relevance rules, junk handling and spatial criteria must be maintained when reproducing the experiments.

The repository distributes model fusion parameters and derived evaluation summaries, not third-party source imagery or pretrained feature-extractor weights.

# Official upstream code and datasets

This internal staging repository contains no benchmark image bytes or third-party pretrained network weights. Obtain required inputs separately and follow their licenses:

- [GSV-Cities](https://github.com/amaralibey/gsv-cities) — development data and disjoint fit/validation/calibration cities.
- [CosPlace](https://github.com/gmberton/CosPlace) and [R2Former](https://github.com/yanyan-bit/R2Former) — original global retrievers.
- [DISK](https://github.com/cvlab-epfl/disk), [ALIKED](https://github.com/Shiaoming/ALIKED), [LightGlue](https://github.com/cvg/LightGlue) — local matching dependencies.
- [SALAD](https://github.com/serizba/salad) and [BoQ](https://github.com/amaralibey/Bag-of-Queries) — additional retrieval-only evaluations.

The protected benchmarks' original definitions, downloads, and distribution rights must be verified from the frozen protocol. Tokyo 24/7 has 147 queries, 16,140 gallery images and declared SUTD ok/junk handling. Only per-query derived metrics, not source images, are present. This repository does not grant any upstream redistribution rights.

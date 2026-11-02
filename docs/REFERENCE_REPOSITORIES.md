# Repository design references

External projects consulted for public documentation and reproducibility *patterns*, without source-code copying:

- [hloc](https://github.com/cvg/Hierarchical-Localization) — localization pipeline organization and data dependencies.
- [LightGlue](https://github.com/cvg/LightGlue) — reproducible matching setup and inference examples.
- [SALAD](https://github.com/serizba/salad) — pretrained retriever usage and evaluation guidance.
- [BoQ](https://github.com/amaralibey/Bag-of-Queries) — pretrained representation and benchmark reporting.

Prior research repositories such as \`saed-alqaraleh/absa-robustness-interference\` and \`saed-alqaraleh/when-to-adapt-speech-deepfake-detector\` use tiered verification separating saved-result checks from full model inference. MPEF similarly distinguishes byte-integrity checks, saved-query numerical replay and unperformed full end-to-end reruns, and additionally checks the frozen 21-feature contract.

"""Small synthetic-input smoke test. Does NOT reproduce empirical results."""
import argparse
from pathlib import Path
from mpef_vpr.synthetic import make_synthetic_batch
from mpef_vpr.features import build_feature_matrix
from mpef_vpr.frozen import load_frozen_model, rank_frozen_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('disk', 'aliked'), default='disk')
    args = parser.parse_args()
    home = Path(__file__).resolve().parents[1]
    model = load_frozen_model(home / 'models', args.backend)
    batch = make_synthetic_batch(queries=2, candidates=100, max_inliers=8, seed=42)
    x, names = build_feature_matrix(batch, expected_k=100)
    ranking, probability = rank_frozen_candidates(model, x, batch.global_rank, batch.candidate_ids)
    print('Software-only demonstration; synthetic observations, NOT benchmark evidence')
    print('Backend:', args.backend, 'Feature count:', len(names))
    for q, order in enumerate(ranking):
        i = int(order[0])
        print(str(batch.query_ids[q]), str(batch.candidate_ids[q, i]), 'calibrated P=',round(float(probability[q, i]), 6))

if __name__ == '__main__':
    main()

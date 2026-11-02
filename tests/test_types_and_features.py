import numpy as np
import unittest

from mpef_vpr.features import FEATURE_SPECS, build_feature_matrix
from mpef_vpr.synthetic import make_synthetic_batch


class TypesAndFeaturesTest(unittest.TestCase):
    def test_batch_and_feature_contract(self):
        batch = make_synthetic_batch(8, 10, max_inliers=8, seed=1)
        self.assertEqual(batch.validate(expected_k=10), (8, 10, 8))
        x, names = build_feature_matrix(batch, expected_k=10)
        self.assertEqual(x.shape, (80, len(FEATURE_SPECS)))
        self.assertEqual(names, tuple(spec.name for spec in FEATURE_SPECS))
        self.assertTrue(np.isfinite(x).all())

    def test_wrong_fixed_k_is_rejected(self):
        batch = make_synthetic_batch(4, 8, max_inliers=5, seed=2)
        with self.assertRaisesRegex(ValueError, "K=100"):
            build_feature_matrix(batch, expected_k=100)

    def test_positive_candidates_have_stronger_synthetic_geometry_on_average(self):
        batch = make_synthetic_batch(20, 10, max_inliers=10, seed=3)
        x, names = build_feature_matrix(batch, expected_k=10)
        y = batch.flat_labels.astype(bool)
        for feature in ["log1p_inliers", "joint_hull_coverage", "joint_rank_support"]:
            j = names.index(feature)
            self.assertGreater(x[y, j].mean(), x[~y, j].mean())

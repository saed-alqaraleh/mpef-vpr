import numpy as np
import unittest

from mpef_vpr.features import build_feature_matrix
from mpef_vpr.model import MPEFModel
from mpef_vpr.ranking import rank_candidates
from mpef_vpr.synthetic import make_synthetic_batch


class IntegrationTest(unittest.TestCase):
    def test_end_to_end_software_path(self):
        q, k = 48, 12
        batch = make_synthetic_batch(q, k, max_inliers=8, seed=6)
        x, _ = build_feature_matrix(batch, expected_k=k)
        y, groups = batch.flat_labels, batch.query_groups

        def rows(start, stop):
            qi = np.arange(start, stop)
            return (qi[:, None] * k + np.arange(k)[None, :]).reshape(-1)

        dev, val, cal = rows(0, 24), rows(24, 36), rows(36, 48)
        model, _ = MPEFModel.select_on_validation(
            x[dev], y[dev], groups[dev], x[val], y[val], groups[val],
            l1_grid=(1e-3,), l2_grid=(1e-2,),
        )
        model.fit_calibrator(x[cal], y[cal], groups[cal])
        p = model.predict_proba(x).reshape(q, k)
        order = rank_candidates(p, batch.global_rank, batch.candidate_ids)
        self.assertEqual(p.shape, (q, k))
        self.assertEqual(order.shape, (q, k))
        self.assertTrue(all(sorted(row.tolist()) == list(range(k)) for row in order))

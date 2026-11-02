import numpy as np
import tempfile
import unittest
from pathlib import Path

from mpef_vpr.features import FEATURE_SPECS, build_feature_matrix
from mpef_vpr.model import MPEFModel
from mpef_vpr.synthetic import make_synthetic_batch


def _split_rows(q, k):
    return (np.asarray(q)[:, None] * k + np.arange(k)[None, :]).reshape(-1)


class ModelTest(unittest.TestCase):
    def test_fit_select_calibrate_and_round_trip(self):
        batch = make_synthetic_batch(60, 10, max_inliers=8, seed=4)
        x, _ = build_feature_matrix(batch, expected_k=10)
        y, groups = batch.flat_labels, batch.query_groups
        dev, val, cal = _split_rows(range(30), 10), _split_rows(range(30, 45), 10), _split_rows(range(45, 60), 10)
        model, records = MPEFModel.select_on_validation(
            x[dev], y[dev], groups[dev], x[val], y[val], groups[val],
            l1_grid=(0.0, 1e-3), l2_grid=(1e-2,),
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(np.all(model.coefficients >= 0))
        model.fit_calibrator(x[cal], y[cal], groups[cal])
        p = model.predict_proba(x)
        self.assertTrue(np.all((0 <= p) & (p <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = MPEFModel.load(path)
            np.testing.assert_allclose(restored.predict_proba(x), p, atol=1e-12)

    def test_declared_monotonic_directions_are_preserved(self):
        rng = np.random.default_rng(5)
        n, f = 500, len(FEATURE_SPECS)
        x = rng.normal(size=(n, f))
        latent = x[:, 0] - x[:, 3] + 0.5 * x[:, 4]
        y = (latent + rng.normal(scale=0.6, size=n) > 0).astype(int)
        groups = np.repeat(np.arange(100), 5)
        model = MPEFModel(l1=0.0, l2=1e-2).fit(x, y, groups)
        base = np.zeros((2, f))
        for j, spec in enumerate(FEATURE_SPECS):
            pair = base.copy()
            if spec.direction == "increasing":
                pair[:, j] = [-2.0, 2.0]
            else:
                pair[:, j] = [2.0, -2.0]
            p = model.predict_uncalibrated(pair)
            self.assertGreaterEqual(p[1] + 1e-12, p[0])

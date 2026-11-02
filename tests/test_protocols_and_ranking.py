import json
import unittest
from pathlib import Path
import numpy as np
from mpef_vpr.protocols import select_primary_protocol_rows
from mpef_vpr.ranking import rank_candidates

class ProtocolsAndRankingTest(unittest.TestCase):
    def test_deterministic_tie_break(self):
        probability = np.array([0.8, 0.8, 0.8, 0.2])
        global_rank = np.array([2, 1, 1, 4])
        ids = np.array(["b", "z", "a", "d"])
        self.assertEqual(rank_candidates(probability, global_rank, ids).tolist(), [2, 1, 0, 3])

    def test_typed_protocol_selection_avoids_nan_thresholds(self):
        rows = [
            {"dataset_id": "metric", "protocol_id": "m25", "query_id": "q1", "method_id": "a"},
            {"dataset_id": "pair", "protocol_id": "pair_native", "query_id": "q2", "method_id": "a"},
            {"dataset_id": "pair", "protocol_id": "wrong", "query_id": "q2", "method_id": "a"},
        ]
        selected = select_primary_protocol_rows(rows, {"metric": "m25", "pair": "pair_native"})
        self.assertEqual([r["protocol_id"] for r in selected], ["m25", "pair_native"])

    def test_release_feature_contract_is_exact_and_complete(self):
        from mpef_vpr.features import FEATURE_SPECS
        contract=json.loads(Path("configs/feature_contract.json").read_text())
        assert len(contract["features"])==21
        assert [(f["name"],f["direction"]) for f in contract["features"]] == [(spec.name,spec.direction) for spec in FEATURE_SPECS]

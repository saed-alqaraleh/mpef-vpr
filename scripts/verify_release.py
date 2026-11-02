"""Verified evidence-only numerical replay and frozen-artifact integrity checks."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    'disk_model.json':'cbd869f2a7fb2d296faab99ed202593e564446fa674fdc3df15ef924939ca161',
    'aliked_model.json':'43352cdf2d98b4e0c3b0f3babdd0901ba5011bb89a629a0fd56d6039a3b8a983',
    'disk_calibrator.json':'efd681a2e5cfdb0bdb08aa3b7d0e32f1c8fd586117cd2ef6c90d2a5c4645217c',
    'aliked_calibrator.json':'8fab2bc754ae92a6c27b1ac21fb75bfecd7f2af74979d6f8eba67aed4e8a4b21',
}

def verify():
    for name,sha in EXPECTED.items():
        path=ROOT/'models'/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=sha: raise AssertionError(f'{name} SHA-256 mismatch')
        print('PASS frozen bytes:',name)
    final=json.loads((ROOT/'results/modern_retrieval/FINAL_TOKYO_COMPLETION.json').read_text())
    assert final['status']=='PASS'
    for record in final['results']:
        backend=record['model']
        with (ROOT/f'results/modern_retrieval/tokyo_{backend}_metrics_by_query.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        if len(rows)!=147:raise AssertionError('Tokyo must have 147 physical queries')
        if len({r['query_id'] for r in rows})!=147:raise AssertionError('Duplicate Tokyo query IDs')
        for metric,col in [('recall_at_1','recall_at_1'),('recall_at_5','recall_at_5'),('recall_at_10','recall_at_10'),('map_all_positives_at_100','map_all_positives_at_100')]:
            mean=sum(float(r[col]) for r in rows)/len(rows)
            if abs(mean-float(record[metric]))>1e-10:raise AssertionError(f'Tokyo {backend} {metric}: {mean} differs')
        assert all(r['dataset']=='tokyo24/7' and r['model']==backend for r in rows)
        print('PASS Tokyo retrieval-only numerical replay:',backend,'147 queries; R@1=',f'{100*record["recall_at_1"]:.2f}%')
    print('PASS: scoped evidence audit. No six-benchmark end-to-end replay was attempted.')
if __name__=='__main__':
    try: verify()
    except Exception as exc:
        print('FAIL:',exc,file=sys.stderr);sys.exit(1)

"""Score independently generated, schema-compliant raw evidence arrays."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from mpef_vpr.frozen import load_frozen_model, rank_frozen_candidates
from mpef_vpr.features import build_feature_matrix
from mpef_vpr.io import load_candidate_batch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=Path,required=True)
    ap.add_argument('--backend',choices=('disk','aliked'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--k',type=int,choices=(50,100),default=100)
    args=ap.parse_args()
    # K=50 must be independently extracted from a K=50 list, not sliced from K=100 feature rows.
    batch=load_candidate_batch(args.evidence,expected_k=args.k)
    x,_=build_feature_matrix(batch,expected_k=args.k)
    model=load_frozen_model(Path(__file__).resolve().parents[1]/'models',args.backend)
    order,probs=rank_frozen_candidates(model,x,batch.global_rank,batch.candidate_ids)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(['query_id','position','candidate_id','probability'])
        for qi,indices in enumerate(order):
            for position,j in enumerate(indices,1):
                w.writerow([batch.query_ids[qi],position,batch.candidate_ids[qi,j],f'{probs[qi,j]:.12g}'])
    print(f'Scored {len(order)} queries with K={args.k}; calibration applied only to probability output')

if __name__=='__main__': main()

"""Release-scoped tests against byte-verified frozen scientific models."""
from pathlib import Path
import numpy as np
import pytest
from mpef_vpr.frozen import load_frozen_model, rank_frozen_candidates
from mpef_vpr.features import FEATURE_SPECS
from mpef_vpr.synthetic import make_synthetic_batch
from mpef_vpr.features import build_feature_matrix

HOME=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('backend', ['disk','aliked'])
def test_frozen_model_and_calibration(backend):
    model=load_frozen_model(HOME/'models',backend)
    assert len(model.feature_specs)==21
    assert all(s.name==a.name and s.direction==a.direction for s,a in zip(model.feature_specs,FEATURE_SPECS))
    assert np.all(model.coefficients>=0)
    assert model.calibrator is not None
    rng=np.random.default_rng(11)
    x=rng.normal(size=(9,21))
    p=model.predict_proba(x)
    assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
    for j,spec in enumerate(FEATURE_SPECS):
        base=x[:1].copy();better=base.copy();better[0,j]+=1.0 if spec.direction=='increasing' else -1.0
        assert model.predict_uncalibrated(better)[0] + 1e-12 >= model.predict_uncalibrated(base)[0]
        assert model.predict_proba(better)[0]+1e-12>=model.predict_proba(base)[0]

@pytest.mark.parametrize('backend', ['disk','aliked'])
def test_real_frozen_model_synthetic_input_end_to_end(backend):
    batch=make_synthetic_batch(queries=2,candidates=50,max_inliers=8,seed=42)
    x,names=build_feature_matrix(batch,expected_k=50)
    model=load_frozen_model(HOME/'models',backend)
    ranks,p=rank_frozen_candidates(model,x,batch.global_rank,batch.candidate_ids)
    assert ranks.shape==(2,50) and p.shape==(2,50)
    assert all(len(set(row))==50 for row in ranks)

def test_corrupt_model_rejected(tmp_path):
    for name in ('disk_model.json','disk_calibrator.json'):
        (tmp_path/name).write_bytes((HOME/'models'/name).read_bytes())
    (tmp_path/'disk_model.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_frozen_model(tmp_path,'disk')

def test_separate_k_feature_reconstruction():
    batch=make_synthetic_batch(queries=1,candidates=100,max_inliers=8,seed=28)
    x100,names=build_feature_matrix(batch,expected_k=100)
    from mpef_vpr.io import subset_queries
    from mpef_vpr.types import CandidateBatch
    vals={k: v for k,v in batch.__dict__.items()}
    for k,v in vals.copy().items():
        if k not in ('query_ids','query_width','query_height') and v is not None:
            vals[k]=v[:, :50] if getattr(v,'ndim',0)>=2 else v
    vals['global_rank']=np.arange(1,51)[None,:]
    subset=CandidateBatch(**vals)
    x50,_=build_feature_matrix(subset,expected_k=50)
    j=names.index('global_rank_percentile')
    assert not np.allclose(x50[:,j],x100[:50,j])

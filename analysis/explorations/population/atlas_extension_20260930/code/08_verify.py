"""Focused reproducibility checks and output inventory for retained conclusions."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    checks={}
    for item in json.loads((OUT/'logs/input_manifest.json').read_text()):
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    checks['inputs_unchanged']=True
    for path in (OUT/'code').glob('*.py'):
        compile(path.read_text(),str(path),'exec')
    compile((OUT/'run_analysis.py').read_text(),str(OUT/'run_analysis.py'),'exec')
    checks['python_sources_compile']=True
    # Existing chemical-class fits, interpreted only on the same neural groups.
    spec=importlib.util.spec_from_file_location('chemical_profiles',OUT/'code/03_chemical_profiles.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_outputs();module.interpret_bridge()
    checks['chemical_class_checks_and_contribution_reconstruction']='passed'
    p=pd.read_csv(OUT/'tables/targeted_chemistry_predictions.csv')
    key=['model','neuron_class','window','date','worm_key','sample_id']
    assert not p.duplicated(key).any()
    assert np.isfinite(p[['observed','predicted','weight']]).all().all()
    assert p.groupby(['model','neuron_class','window','sample_id']).weight.sum().sub(1).abs().max()<1e-12
    a=pd.read_csv(ROOT/'reports/exploration_20260929/tables/animal_metrics.csv')
    repetitions=a.groupby('sample_id').date.nunique();repeats=set(repetitions[repetitions>1].index)
    assert not repeats.intersection(p.sample_id)
    for held in p.date.unique():
        train=a[(a.date!=held)&~a.sample_id.isin(repeats)]
        test=p[p.date==held]
        assert not(set(train.sample_id)&set(test.sample_id))
        assert not(set(zip(train.date,train.worm_key))&set(zip(test.date,test.worm_key)))
    checks['chemical_split_weight_and_finite_predictions']='passed'
    risk=pd.read_csv(OUT/'tables/targeted_chemistry_inner_risks.csv')
    group=['outer_block','model','neuron_class','window','penalty']
    means=risk.groupby(group).mse.mean().reset_index()
    selected=means.loc[means.groupby(group[:-1]).mse.idxmin()]
    stored=pd.read_csv(OUT/'tables/targeted_chemistry_choices.csv')
    q=selected.merge(stored,on=group[:-1],suffixes=('_new','_saved'),validate='one_to_one')
    assert len(q)==1404 and np.all(q.penalty_new==q.penalty_saved)
    checks['nested_parameter_choices_recomputed']=len(q)
    actual=p.groupby(['model','neuron_class','window','sample_id'])[['observed','predicted']].mean().reset_index()
    expected=pd.read_csv(OUT/'tables/targeted_chemistry_scores.csv')
    errors=[]
    for row in expected.itertuples():
        d=actual[(actual.model==row.model)&(actual.neuron_class==row.neuron_class)&(actual.window==row.window)]
        r2=1-np.sum((d.observed-d.predicted)**2)/np.sum(d.observed**2)
        errors.append(abs(r2-row.r2_strain_mean))
    assert max(errors)<1e-12
    checks['maximum_strain_r2_recalculation_error']=max(errors)
    curves=pd.read_csv(OUT/'tables/response_patterns_matched_five_curves.csv')
    assert curves.groupby('neuron_class').size().eq(64).all()
    assert len(curves[['date','worm_key']].drop_duplicates())==32
    membership=pd.read_csv(OUT/'tables/population_bridge_membership.csv')
    assert len(membership[['date','worm_key']].drop_duplicates())==16
    assert membership.date.nunique()==4 and membership.sample_id.nunique()==33
    assert membership.groupby(['date','worm_key','group']).sample_id.nunique().eq(3).all()
    bridge=pd.read_csv(OUT/'tables/population_bridge_animal_contrasts.csv')
    assert bridge.groupby(['model','neuron_class']).size().eq(16).all()
    # Direct independent reconstruction from stored curves, without model code.
    raw=pd.read_parquet(ROOT/'reports/exploration_20260929/tables/animal_curves.parquet').reset_index()
    raw['date']=raw.date.astype(int);raw['post']=raw[[str(t) for t in range(10,30)]].mean(axis=1)
    match=raw.merge(membership[['date','worm_key','sample_id','group']],on=['date','worm_key','sample_id'])
    m=match.groupby(['date','worm_key','neuron_class','group']).post.mean().unstack('group')
    m['direct_difference']=m.higher-m.lower
    v=bridge[bridge.model=='panel162'].merge(m[['direct_difference']],left_on=['date','worm_key','neuron_class'],right_index=True,validate='one_to_one')
    assert len(v)==80 and (v.observed-v.direct_difference).abs().max()<1e-12
    checks['bridge_original_curve_error']=float((v.observed-v.direct_difference).abs().max())
    checks['main_curve_animals']=32;checks['chemical_bridge_animals']=16
    missing=[]
    for doc in ['REVIEW.md','REPORT.md','README.md','RESEARCH_LOG.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(OUT/doc).read_text()):
            if target.startswith(('http:','https:','#')):continue
            path=OUT/target.split('#')[0]
            if not path.exists():missing.append([doc,target])
    # This file itself and final inventory are written below.
    missing=[x for x in missing if x[1] not in ['logs/verification.json','logs/output_manifest.json']]
    assert not missing,missing
    checks['report_links']='passed'
    (OUT/'logs/verification.json').write_text(json.dumps(dict(status='passed',checks=checks,
        interpretation='Arithmetic, lineage and output checks; not biological independent validation.'),indent=2))
    files=[]
    for path in sorted(OUT.rglob('*')):
        if not path.is_file() or '__pycache__' in str(path):continue
        # Streaming log files can still change after this process prints its
        # result; inventory stable artifacts, not partially written stdout.
        if path.suffix=='.log' or path.name in ['output_manifest.json','run_status.json']:continue
        files.append(dict(path=str(path.relative_to(OUT)),bytes=path.stat().st_size,sha256=sha(path)))
    (OUT/'logs/output_manifest.json').write_text(json.dumps(files,indent=2))
    print(json.dumps(dict(status='passed',checks=checks,files_recorded=len(files)),indent=2))


if __name__=='__main__':main()

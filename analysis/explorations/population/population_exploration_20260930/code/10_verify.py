"""Numerical, design and delivery checks for this completed research round."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
OLD=ROOT/'reports/exploration_20260929/tables'


def main():
    checks={}
    (OUT/'logs/verification.json').write_text(json.dumps({'status':'running'}))
    manifest=json.loads((OUT/'logs/input_manifest.json').read_text())
    for record in manifest:
        path=ROOT/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path']
    checks['input_hashes_unchanged']=len(manifest)

    # Independently reconstruct the 16 calcium curves used in the new focused
    # within-genus figure from the source parquet rather than exported means.
    columns=['date','worm_key','segment_index','neuron','time_point','delta_F_over_F0','stim_name']
    raw=pd.read_parquet(ROOT/'data/106bac.parquet',columns=columns)
    raw['sample_id']=raw.stim_name.str.extract(r'^(A\d{3})',expand=False)
    sub=raw[raw.sample_id.isin(['A178','A189'])&raw.neuron.isin(['ADFL','ADFR','AWAL','AWAR'])].copy()
    sub['neuron_class']=sub.neuron.str[:-1]
    sub['date']=sub.date.astype(str)
    keys=['sample_id','date','worm_key','neuron_class']
    trial=sub.groupby(keys+['segment_index','time_point']).delta_F_over_F0.mean()
    animal=trial.groupby(keys+['time_point']).mean().unstack('time_point')
    animal.columns=[str(int(t)-5) for t in animal.columns]
    fig=pd.read_csv(OUT/'tables/figure_bifidobacterium_curves.csv',dtype={'date':str}).set_index(keys)
    assert len(animal)==len(fig)==16
    error=np.max(np.abs(animal.to_numpy()-fig.reindex(animal.index)[animal.columns].to_numpy()))
    assert error<1e-12
    checks['source_parquet_figure_curves']={'n':16,'max_abs_error':float(error)}
    del raw,sub,trial

    pred=pd.read_csv(OUT/'tables/information_predictions.csv')
    summary=pd.read_csv(OUT/'tables/information_summary.csv')
    for row in summary.itertuples():
        d=pred[pred.window.eq(row.window)&pred['mode'].eq(row.mode)]
        acc=d.groupby(['date','worm_key']).correct.mean().mean()
        assert abs(acc-row.accuracy)<1e-12
        assert d[['date','worm_key']].drop_duplicates().shape[0]==49
        assert len(d)==607
        assert d.groupby(['date','worm_key']).apply(lambda z:np.allclose(z.chance,1/z.sample_id.nunique()),include_groups=False).all()
    checks['information_summary_recomputed']=len(summary)

    spec=importlib.util.spec_from_file_location('information',OUT/'code/05_population_information.py')
    info=importlib.util.module_from_spec(spec);spec.loader.exec_module(info)
    rng=np.random.default_rng(2026093010)
    train=rng.normal(size=(4,12,13));test=rng.normal(size=(12,13));labels=[f's{i}' for i in range(12)]
    base=info.predict(train,test,labels,np.arange(13),'raw_shape')
    scaled=info.predict(train,test*np.exp(rng.normal(size=(12,1))),labels,np.arange(13),'raw_shape')
    assert np.array_equal(base,scaled)
    checks['raw_shape_test_gain_invariance']=True

    a=pd.read_csv(OLD/'animal_metrics.csv')
    repeated=a.groupby('sample_id').date.nunique();excluded=set(repeated[repeated>1].index)
    chemistry=pd.read_csv(OUT/'tables/chemistry_population_predictions.csv')
    assert chemistry.sample_id.nunique()==100 and not(set(chemistry.sample_id)&excluded)
    assert chemistry[['date','worm_key']].drop_duplicates().shape[0]==49
    group=['model','lambda','window','neuron_class','date']
    assert np.allclose(chemistry.groupby(group+['sample_id']).weight.sum(),1)
    assert np.allclose(chemistry.groupby(group+['worm_key'])[['observed','predicted']].mean(),0,atol=1e-12)
    chemistry['sse']=chemistry.weight*((chemistry.observed-chemistry.predicted)/chemistry.train_scale)**2
    chemistry['sst']=chemistry.weight*(chemistry.observed/chemistry.train_scale)**2
    calc=chemistry.groupby(['model','window'])[['sse','sst']].sum()
    calc['relative_r2']=1-calc.sse/calc.sst
    saved=pd.read_csv(OUT/'tables/chemistry_population_summary.csv').set_index(['model','window'])
    assert np.allclose(calc.relative_r2,saved.reindex(calc.index).relative_r2,atol=1e-12)
    checks['chemistry_centering_weights_scores']=True
    # Every training complement of a held-out block is disjoint in animals
    # and in strains after removing the six repeated strains.
    unique=a[~a.sample_id.isin(excluded)][['date','worm_key','sample_id']].drop_duplicates()
    for date in unique.date.unique():
        tr=unique[~unique.date.eq(date)];te=unique[unique.date.eq(date)]
        assert not(set(tr.sample_id)&set(te.sample_id))
        assert not(set(zip(tr.date,tr.worm_key))&set(zip(te.date,te.worm_key)))
    checks['chemical_fold_disjointness']=9

    neighbors=pd.read_csv(OUT/'tables/chem_neighbors_members.csv')
    checks['chemical_neighbor_rows']=len(neighbors)
    assert len(neighbors)==600
    assert neighbors.held_block.ne(neighbors.neighbor_block).all()
    assert not(set(neighbors.sample_id)&excluded) and not(set(neighbors.neighbor)&excluded)
    assert neighbors.groupby(['model','sample_id']).neighbor.nunique().eq(3).all()
    checks['chemical_neighbors_disjoint']=True

    files=['REVIEW.md','REPORT.md','README.md','RESEARCH_LOG.md']
    broken=[]
    for f in files:
        content=(OUT/f).read_text()
        for target in re.findall(r'\]\(([^)]+)\)',content):
            if target.startswith(('https://','http://','#')):
                continue
            path=target.split('#')[0]
            if not (OUT/path).exists():
                broken.append(f'{f}: {target}')
    assert not broken,broken
    checks['document_local_links_valid']=True
    figures=['explore_population_combinations','02_within_genus_composition','S01_population_information','explore_temporal_bifidobacterium']
    for name in figures:
        for ext in ['png','pdf']:
            assert (OUT/'figures'/f'{name}.{ext}').stat().st_size>1000
    checks['figure_files_present']=8
    checks['status']='passed'
    checks['interpretation']='Numerical and design checks passed; this is not independent biological validation.'
    (OUT/'logs/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()

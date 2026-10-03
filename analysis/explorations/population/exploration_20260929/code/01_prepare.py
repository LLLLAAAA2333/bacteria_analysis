"""Prepare auditable animal curves; never writes to source data or notebooks.

Run with the project's .pixi/envs/default/bin/python from any directory.
One volume is one second (01 notebook cell42); time zero below is stimulus onset.
Stored dF/F0 uses upstream fitted F0, not recomputed here. Available bilateral
channels are averaged within a trial, then repeated trials within each animal.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
TABLES = OUT / 'tables'
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
BILATERAL = NEURONS[:9]
KEY = ['sample_id','date','worm_key','neuron_class']
SEED = 20260929
WINDOWS = {'stim': (0,10), 'post': (10,30), 'late': (30,40), 'full': (0,40)}


def main():
    TABLES.mkdir(exist_ok=True)
    raw = pd.read_parquet(ROOT / 'data/106bac.parquet')
    assert raw.start_time.eq(5).all() and raw.end_time.eq(15).all()
    assert set(raw.time_point) == set(range(45))
    raw['sample_id'] = raw.stim_name.str.extract(r'^(A\d{3})', expand=False)
    assert raw.sample_id.notna().all()
    assert not raw.duplicated(['date','worm_key','segment_index','neuron','time_point']).any()
    assert raw.groupby(['date','worm_key','segment_index']).sample_id.nunique().eq(1).all()
    raw['neuron_class'] = raw.neuron.replace({n+s:n for n in BILATERAL for s in ['L','R']})
    d = raw[raw.neuron_class.isin(NEURONS)].copy()
    assert np.isfinite(d.delta_F_over_F0).all()
    tkey = KEY + ['segment_index','time_point']
    trial = d.groupby(tkey, observed=True).delta_F_over_F0.mean().unstack('time_point')
    side_counts = d.groupby(tkey, observed=True).size().unstack('time_point')
    assert trial.notna().all().all()
    trial.columns = [str(i-5) for i in trial.columns]
    trial.to_parquet(TABLES/'trial_curves.parquet')
    animal = trial.groupby(level=KEY).mean()
    animal.to_parquet(TABLES/'animal_curves.parquet')
    metric = animal.index.to_frame(index=False)
    for name,(a,b) in WINDOWS.items():
        metric[name] = animal[[str(i) for i in range(a,b)]].mean(axis=1).to_numpy()
    metric['post_minus_stim'] = metric.post - metric.stim
    metric['baseline'] = animal[[str(i) for i in range(-5,0)]].mean(axis=1).to_numpy()
    metric['baseline_sd'] = trial[[str(i) for i in range(-5,0)]].std(axis=1).groupby(level=KEY).median().to_numpy()
    metric['n_trials'] = trial.groupby(level=KEY).size().to_numpy()
    metric['bilateral_complete'] = side_counts.eq(2).all(axis=1).groupby(level=KEY).all().to_numpy()
    metric.to_csv(TABLES/'animal_metrics.csv',index=False)
    bins = pd.DataFrame({str(i):animal[[str(j) for j in range(i,i+5)]].mean(axis=1) for i in range(0,40,5)})
    bins.to_csv(TABLES/'animal_bins.csv')
    date = metric.groupby(['sample_id','date','neuron_class']).agg(
        n_animals=('worm_key','nunique'), n_trials=('n_trials','sum'),
        **{w:(w,'mean') for w in [*WINDOWS,'post_minus_stim','baseline']})
    date.to_csv(TABLES/'date_metrics.csv')
    strain = date.groupby(['sample_id','neuron_class'])[[*WINDOWS,'post_minus_stim','baseline']].mean()
    strain.to_csv(TABLES/'strain_metrics.csv')
    coverage = raw[['sample_id','date','worm_key','segment_index']].drop_duplicates()
    coverage.to_csv(TABLES/'trial_design.csv',index=False)
    pd.crosstab(coverage.sample_id,coverage.date).to_csv(TABLES/'strain_date_trial_counts.csv')
    counts = metric.groupby(['sample_id','date','neuron_class']).size().unstack('neuron_class').reindex(columns=NEURONS)
    counts.to_csv(TABLES/'coverage.csv')
    repeat = coverage.groupby('sample_id').date.nunique()
    audit = dict(raw_rows=len(raw), retained_rows=len(d), neurons=NEURONS,
                 strains=raw.sample_id.nunique(), dates=raw.date.nunique(),
                 animals=raw[['date','worm_key']].drop_duplicates().shape[0],
                 trials=len(coverage), animal_strain_pairs=coverage[['sample_id','date','worm_key']].drop_duplicates().shape[0],
                 animal_neuron_curves=len(animal), missing_class_date_cells=int(counts.isna().sum().sum()),
                 min_animals_per_class_date=int(counts.min().min()),max_animals_per_class_date=int(counts.max().max()),
                 repeated_strains=repeat[repeat>1].to_dict(),
                 baseline_absolute_quantiles=metric.baseline.abs().quantile([.5,.9,.99,1]).to_dict(),
                 trials_per_animal_strain_quantiles=coverage.groupby(['sample_id','date','worm_key']).size().quantile([0,.5,1]).to_dict())
    (OUT/'logs/data_audit.json').write_text(json.dumps(audit,indent=2))
    inputs = [ROOT/'data'/n for n in ['106bac.parquet','matrix.xlsx','metabolism_raw_data.xlsx',
              'GM300_bacteria_species_summary.xlsx','current_samples.xlsx']]
    inputs += [ROOT/'notebook'/n for n in ['01_measurement_first_inspection.ipynb','02_reproducibility_inspection.ipynb']]
    inputs += [ROOT/'pixi.toml',ROOT/'pixi.lock',ROOT/'reports/neural_chemical_restart_20260929.md']
    manifest = [{'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
    manifest_path = OUT/'logs/input_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Source files changed since this round began'
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2))
    environment = dict(python=sys.version, executable=sys.executable,platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','pyarrow','scikit-learn','openpyxl','statsmodels']},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status_at_preparation=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        seed=SEED,windows_seconds_after_onset=WINDOWS)
    (OUT/'logs/environment.json').write_text(json.dumps(environment,indent=2))
    print(json.dumps(audit,indent=2))
    print('\nStrain response distribution (equal dates):')
    print(strain.groupby('neuron_class')[['stim','post','full','post_minus_stim']].agg(['min','median','max']).round(3).to_string())


if __name__ == '__main__':
    main()

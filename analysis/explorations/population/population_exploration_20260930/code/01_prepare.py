"""Record the inputs reused from the verified earlier processing round.

This round does not re-extract fluorescence or rerun historical analyses.
All source data and previous outputs are read-only. The unit of neural sampling
is an animal identified by (date, worm_key); repeated trials are averaged.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'reports/exploration_20260929'


def main():
    inputs = [PREV / 'tables' / n for n in [
        'animal_metrics.csv', 'animal_curves.parquet', 'trial_curves.parquet',
        'trial_design.csv', 'taxonomy.csv', 'chemical_raw.csv', 'chemical_log.csv',
        'chemical_feature_metadata.csv', 'chemical_reference_groups.csv',
        'chemical_legacy_logfc.csv', 'strain_metrics.csv', 'date_metrics.csv']]
    inputs += [PREV / 'code' / n for n in ['01_prepare.py', '03_chemical_prepare.py']]
    inputs += [PREV / 'logs' / n for n in ['input_manifest.json', 'data_audit.json',
                                        'chemical_audit.json', 'verification.json']]
    inputs += [ROOT / 'data' / n for n in ['106bac.parquet', 'metabolism_raw_data.xlsx',
        'matrix.xlsx', 'GM300_bacteria_species_summary.xlsx', 'current_samples.xlsx']]
    inputs += [ROOT / 'pixi.toml', ROOT / 'pixi.lock', ROOT / 'AGENTS.md']
    records = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
    target = OUT / 'logs/input_manifest.json'
    if target.exists():
        assert json.loads(target.read_text()) == records, 'An analysis input has changed'
    else:
        target.write_text(json.dumps(records, indent=2))
    old = json.loads((PREV / 'logs/input_manifest.json').read_text())
    for rec in old:
        if rec['path'].startswith('data/'):
            assert hashlib.sha256((ROOT / rec['path']).read_bytes()).hexdigest() == rec['sha256']
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               packages={p: importlib.metadata.version(p) for p in
                         ['numpy','pandas','scipy','matplotlib','scikit-learn','pyarrow']},
               git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               git_status=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
               seed=20260930)
    (OUT / 'logs/environment.json').write_text(json.dumps(env, indent=2))
    print(f'Verified {len(records)} inputs; original data unchanged from prior audited round.')


if __name__ == '__main__':
    main()

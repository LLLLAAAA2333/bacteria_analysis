"""Record local inputs and runtime; raw and earlier outputs are read-only."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    prior=json.loads((ROOT/'reports/population_exploration_20260930/logs/input_manifest.json').read_text())
    entries=[]
    for item in prior:
        p=ROOT/item['path']
        actual=sha(p)
        assert actual==item['sha256'],f'Prior verified input changed: {p}'
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=actual))
    added=['reports/population_exploration_20260930/tables/information_predictions.csv',
           'reports/population_exploration_20260930/tables/chemical_groups_members.csv',
           'reports/population_exploration_20260930/tables/chemistry_population_predictions.csv']
    for name in added:
        if name not in [x['path'] for x in entries]:
            p=ROOT/name;entries.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p)))
    (OUT/'logs/input_manifest.json').write_text(json.dumps(entries,indent=2))
    packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','matplotlib','pyarrow','scikit-learn']}
    (OUT/'logs/environment.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,
        platform=platform.platform(),packages=packages,raw_inputs_changed=False,
        input_count=len(entries),scope='New output directory only; inherited processed table provenance independently checked.'),indent=2))
    print(f'{len(entries)} input hashes recorded; all inherited inputs unchanged.')


if __name__=='__main__':main()

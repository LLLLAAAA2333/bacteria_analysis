"""Reproduce this round only, leaving original and earlier outputs untouched."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
env=os.environ.copy()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    env[key]='1'
scripts=['00_provenance','01_atlas_roles','02_response_patterns','04_targeted_chemistry',
         '03_chemical_profiles','05_chemical_checks','06_within_genus_chemistry',
         '07_population_bridge','08_verify']
records=[]
for name in scripts:
    start=time.monotonic()
    with (OUT/'logs'/f'run_{name}.log').open('w') as log:
        result=subprocess.run([sys.executable,str(OUT/'code'/f'{name}.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(script=name+'.py',exit_code=result.returncode,seconds=time.monotonic()-start))
    (OUT/'logs/run_status.json').write_text(json.dumps(dict(execution_mode='serial runner',runs=records),indent=2))
    print(name,result.returncode,round(records[-1]['seconds'],2),'seconds',flush=True)
    if result.returncode:raise SystemExit(result.returncode)

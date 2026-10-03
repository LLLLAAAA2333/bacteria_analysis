"""Reproduce this bounded round, writing only under its output directory."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SCRIPTS=['01_prepare.py','02_population_patterns.py','03_temporal_patterns.py',
         '04_chemical_groups.py','05_population_information.py','06_information_checks.py',
         '07_population_chemistry.py','08_chemical_neighbors.py','09_synthesis.py','10_verify.py']


def main():
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    records=[]
    for script in SCRIPTS:
        start=time.monotonic()
        rec=dict(script=script,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 command=[sys.executable,str(OUT/'code'/script)])
        with (OUT/'logs'/f'run_{script[:-3]}.log').open('w') as log:
            process=subprocess.run(rec['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        rec.update(returncode=process.returncode,seconds=round(time.monotonic()-start,3))
        records.append(rec)
        (OUT/'logs/run_status.json').write_text(json.dumps(records,indent=2))
        print(f'{script}: exit {process.returncode}, {rec["seconds"]:.2f} s',flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__=='__main__':
    main()

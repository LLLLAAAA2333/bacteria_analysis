"""Run this round's short, fixed script sequence and save actual exit status."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SCRIPTS=['01_prepare.py','02_neural_overview.py','03_chemical_prepare.py',
         '04_phenotype_checks.py','05_chemical_tests.py','06_asparagine_check.py','07_verify.py']


def main():
    records=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for script in SCRIPTS:
        start=time.monotonic()
        command=[sys.executable,str(OUT/'code'/script)]
        log=OUT/'logs'/('final_'+script.replace('.py','.log'))
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        records.append(dict(script=script,command=command,returncode=result.returncode,
                            elapsed_seconds=round(time.monotonic()-start,3),log=str(log.relative_to(OUT))))
        (OUT/'logs/run_status.json').write_text(json.dumps(records,indent=2))
        print(f'{script}: exit {result.returncode}, {records[-1]["elapsed_seconds"]} s',flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=='__main__':
    main()

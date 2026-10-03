"""Run the existing small scripts sequentially using this Python environment."""
from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    states = []
    for path in sorted((OUT / "code").glob("[0-9][0-9]_*.py")):
        start = datetime.now(timezone.utc).isoformat()
        log = OUT / "logs" / (path.stem + "_replay.log")
        print(f"Running {path.name}; output: {log}", flush=True)
        with log.open("w") as handle:
            result = subprocess.run([sys.executable, str(path)],
                                    stdout=handle, stderr=subprocess.STDOUT)
        states.append(dict(script=path.name, started_utc=start,
                           finished_utc=datetime.now(timezone.utc).isoformat(),
                           exit_code=result.returncode, log=str(log.relative_to(OUT))))
        (OUT / "logs/run_status.json").write_text(json.dumps(states, indent=2))
        if result.returncode:
            raise SystemExit(result.returncode)

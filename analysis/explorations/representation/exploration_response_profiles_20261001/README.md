# Independent response-representation exploration

Read [REPORT.md](REPORT.md) for the findings and limitations. These outputs explore the requested profile → repeatability → neural/chemical RDM sequence. Original notebooks and previous poster figures are not edited.

From the repository root, reproduce this directory with:

```sh
MPLBACKEND=Agg .pixi/envs/default/bin/python reports/exploration_response_profiles_20261001/code/run_exploration.py
```

The runner reads reviewed caches, verifies provenance, fits the authorized full-data representations, performs 100 independent animal-half partitions and 49 animal-held-out fits, and writes this directory's tables and PNG/SVG inspection figures. It overwrites these exploration outputs. It does not run any notebook.

The small functions can also be called from an exploratory Python session:

```python
from pathlib import Path
import sys

root = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
out = root / 'reports/exploration_response_profiles_20261001'
sys.path.insert(0, str(out / 'code'))

from response_representation import load_inputs, fit_representation, aggregate_strains

data = load_inputs(root / 'reports')
# raw: animal × (strain, acquisition date) × cell × 40 one-second samples
# All response values and coefficients are in ΔF/F0; templates have RMS=1.
fit = fit_representation(
    data['raw'], data['baseline_sd'],
    [strain for strain, date in data['conditions']],
    threshold=1.0, min_animals=3,
)
amplitudes, strains = aggregate_strains(fit['coefficients'], data['conditions'])
# amplitudes: 106 strains × 13 cell classes; NaN stays missing, gate failure is 0.
```

`threshold=None` disables the gate. Cutoffs 0.5, 1, 1.5 and 2 are exploratory, not calibrated response-detection thresholds. Templates are refitted at each cutoff; coefficients from different cutoffs therefore need not share the same temporal template. The saved `raw_coefficient` is the projection before zeroing onto that cutoff's template.

Tests:

```sh
.pixi/envs/default/bin/python -m unittest discover -s reports/exploration_response_profiles_20261001/code -p test_response_profiles.py -v
```

The tests use synthetic arrays only. Input provenance, thresholds, missingness rules and hashes are recorded in `representation_parameters.json`; validation and RDM summaries are in `comparison_results.json`; plotting decisions and external captions are under `figures/`.

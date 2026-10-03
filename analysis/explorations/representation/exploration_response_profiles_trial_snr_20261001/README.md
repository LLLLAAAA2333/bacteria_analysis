# Trial-SNR exploration

See [REPORT.md](REPORT.md), [filtering heatmap](figures/00_filter_status_heatmap.png), and [figure captions](figures/captions.txt). This version preserves the previous individual-SNR exploration and all original notebooks.

Code is split into four small modules: representation, comparisons/bootstrap, plotting, and a wrapper around the original HMDS fitter. Cached trial data are read only. Full fitting uses 0–40 s; the displayed profile uses five 5-s bins over 0–25 s.

To reproduce the analysis in a **new output directory**, run from the repository root:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path.cwd()
source = root / 'reports/exploration_response_profiles_trial_snr_20261001'
sys.path.insert(0, str(source / 'code'))
from run_trial_exploration import main
main(root / 'reports/exploration_response_profiles_trial_snr_rerun')
PY
```

This recomputes full representations, 100 animal-half splits, 49 animal-held-out fits, 1,000 whole-animal bootstrap draws, and neural 2D HMDS. The verified chemical 2D fit is reused only after checking numerical identity of its RDM. Choose another fresh output directory for subsequent full runs; existing HMDS results are not overwritten.

The main adjustable values are explicit function arguments: SNR cutoffs and minimum coverage in `save_full_representation`, repeats/seed in `run_comparisons`, draws/coverage in `run_bootstrap_chord`. The current plotting module labels the primary cutoff 1 and original histogram bin width 0.05 explicitly; update labels if changing that cutoff. Trial SNR is a weighted across-trial scatter measure, not a per-trial prestimulus-baseline test.

Synthetic checks:

```sh
.pixi/envs/default/bin/python -m unittest discover -s reports/exploration_response_profiles_trial_snr_20261001/code -p test_trial_profiles.py -v
```

Key audit outputs are `tables/condition_metrics.csv` (date-level decisions and trial counts), `tables/filter_state_by_strain_cell.csv` (the exact filtering map), `tables/display_profile_5bin.csv`, and `hmds/neural_sample_coverage.csv`. HMDS comparison displays the same selected sample IDs in both domains; the chemical coordinates still originate from the verified full-sample fit.

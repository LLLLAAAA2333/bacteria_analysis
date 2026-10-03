# Individual-SNR exploration

Primary setting: animal-mean 0–40 s curves, SNR ≥ 0.5, at least two animals per strain × date × cell. Trial curves are averaged within animals before SNR estimation. Original notebooks and earlier reports are unchanged.

Start with [REPORT.md](REPORT.md), the [filter map](figures/00_filter_status_heatmap.png), or the [figure captions](figures/captions.txt).

## Code

- `individual_representation.py`: individual SNR, templates and coefficients; reuses the reviewed original loader and numerical template implementation.
- `individual_comparisons.py`: independent animal splits, leave-one-animal-out validation, RDMs and whole-animal bootstrap uncertainty.
- `individual_plots.py`: five-bin display, filter maps, density histograms, validation, RDM and HMDS plots.
- `individual_hmds.py`: original notebook HMDS fitter with convergence checks; verified existing chemical fit.
- `individual_hmds_3d.py`: checked neural 3D fit on exactly the existing 2D inputs and coverage, plus verified chemical 3D reuse.
- `individual_hmds_full.py`: current full-sample 2D/3D HMDS, retaining all finite observed pairs and treating bootstrap coverage as QC rather than an exclusion cutoff.
- `individual_profile_display.py`: notebook-style 5-bin profile and separate compression/template/amplitude model figure.
- `individual_comparison_display.py`: RdBu_r RDMs and separate 2D/3D HMDS figures with Shepard diagrams; 3D is shown as XY/XZ/YZ orthographic views of the same saved coordinates. Embedding points reuse the notebook's exact saved turbo colors and full-reference linear normalization.
- `hmds_view_selection.py`: historical single-camera comparison using projected marker overlap only; `run_selection(source)` saves the angle selection and candidate sheet without refitting. The current three-view plotter does not apply this selection.
- `individual_repeatability_display.py`: notebook's short-wide histogram, external legend and original colors; saved data, density normalization and complete tails unchanged.
- `refresh_individual_figures.py`: redraw the requested panels from saved data and checked fits, without response analysis or HMDS refitting.
- `snr_definition_audit.py`: matched arithmetic audit of the saved trial and individual SNR definitions.
- `test_individual_profiles.py`: focused synthetic checks.

## Reproduce in a fresh output directory

Run from the repository root using its `.pixi/envs/default/bin/python`, or execute the following in a notebook using that environment. It reruns the full exploration, including 1000 bootstrap draws and neural HMDS; use it only when a full rerun is intended.

```python
from pathlib import Path
import sys

repo = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
source = repo / 'reports/exploration_response_profiles_individual_snr_20261002'
sys.path.insert(0, str(source / 'code'))
from run_individual_exploration import main

main(repo / 'reports/exploration_response_profiles_individual_snr_rerun')
```

The runner refuses an output directory already containing `hmds/`, `hmds3d/` or `hmds_full106/` to preserve checked results. `PRIMARY_THRESHOLD = 0.5` is defined in `run_individual_exploration.py`; it is passed to fitting, validation, bootstrap and plotting. Sensitivity fits are saved at 0.25, 0.5, 0.75 and 1, plus an unfiltered control. Seed 20261001 preserves the preceding exploration's partition choices.

This directory depends on the reviewed cached inputs and original response-representation module identified in `representation_parameters.json`, plus the notebook HMDS modules. It is not a standalone data package. The report and SNR-definition audit are separate from the main figure runner. The full runner now fits both dimensions on all 106 samples/all 5565 pairs and uses the revised layouts. The original bootstrap 80% mask remains saved for historical QC, but the full-sample fitter does not apply that mask.

For display-only refresh using the existing results, use the same setup above and call:

```python
from refresh_individual_figures import refresh_individual_figures
refresh_individual_figures(source)
```

This overwrites the selected figure files and captions. It requires the saved `hmds_full106/2d/` and `hmds_full106/3d/` checked fits and does not run the full analysis. To build just the new full-sample embeddings from existing D/V tables in a report where `hmds_full106/` does not yet exist, call `individual_hmds_full.run_full_hmds(source, repo)`.

## Saved outputs

- `tables/condition_metrics.csv`: date-level SNR, coverage, decision and amplitude.
- `tables/filter_state_by_strain_cell.csv`: strain-level retained, zeroed, mixed-date and unavailable states.
- `tables/templates.csv`, `tables/strain_coefficients.csv`, `tables/display_profile_5bin.csv`: full templates, amplitudes and displayed reconstruction.
- `tables/sensitivity_metrics.csv`: cutoff comparison.
- `comparison_results.json`: validation and RDM summaries.
- `bootstrap_results.json`, `hmds/neural_sample_coverage.csv`, `hmds/result.json`: bootstrap support and checked embedding diagnostics.
- `snr_definition_audit.json`, `tables/snr_definition_comparison.csv`: why the trial gate removed more entries.
- `verification.json`: input/notebook preservation and code hashes.
- `hmds3d/result.json`, `hmds3d/verification.json`: new neural 3D convergence, units and preservation checks.
- `hmds_full106/{2d,3d}/result.json`: current full-sample fits and diagnostics; `hmds_full106/verification.json` checks the 106 IDs, 5565 pairs, gradients, units and preservation of earlier fits.
- `hmds_full106/neural_sample_coverage.csv`, `neural_pair_coverage_qc.csv`: sample/pair bootstrap QC; lower coverage no longer removes samples, and variance remains conditional on valid draws.
- `figure_revision_verification.json`: figure refresh provenance and unchanged existing analysis inputs.
- `embedding_color_revision_verification.json`: later point-palette correction and exact reference-color checks; supersedes the earlier display hashes.
- `camera_view_verification.json`: earlier single-camera revision, with unchanged input, point-color and Shepard checks.
- `three_view_verification.json`: current XY/XZ/YZ revision; all six plotted coordinate/color arrays and both full-3D Shepard arrays match the saved inputs exactly.

The current 3D figure has neural and chemical rows, with XY/XZ/YZ orthographic projections and a Shepard diagram in each row. All projections use the same saved 3D coordinates, with equal aspect and limits −1.04 to 1.04. These are not separate 2D fits; projected spacing is not hyperbolic distance, and orientations are not aligned across domains. Shepard plots retain the full-3D fitted distances. The prior selected-camera figure and plotting source are archived under `figures/previous_single_view_3d/`; the earlier fixed-camera figure remains under `figures/previous_3d_view/`. Historical [candidate views](figures/3d_view_candidates.png) and [selection record](figures/hmds_3d_view_selection.json) remain available but are not used for the current figure.

Response/model heatmaps and RDMs use RdBu_r. HMDS sample colors use notebook 03's original turbo mapping, including its full 106-sample linear min–max range and exact saved color_hex values. Missing entries remain distinct from gated zeros. Both current HMDS dimensionalities include all 106 samples and all 5565 pairs, matching full RDM coverage. Current figure files are `03b_neural_chemical_hmds_2d` and `03c_neural_chemical_hmds_3d`, each with Shepard panels. The previous 81-sample figures are archived under `figures/previous_81_sample_hmds/`; their numerical results remain in `hmds/` and `hmds3d/`. The earlier `03b_neural_chemical_hmds` is retained as an older display/palette. Previous tall repeatability figures are under `figures/previous_tall_repeatability/`.

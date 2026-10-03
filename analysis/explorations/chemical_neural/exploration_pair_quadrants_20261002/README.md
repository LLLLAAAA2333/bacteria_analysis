# Four distance-corner exploration

The user clarified that the deliverable is an individual-pair PDF, not category
summaries or a finalized Figure 5. The current output is
`output/pdf/neural_chemical_pairs_exploration.pdf` from the repository root,
with `tables/pair_pdf_index.csv` for navigation. Each of the 1,478 pairs has its
own page, showing all 13 neural cells, all 380 chemical features in a scatter,
and the 18 largest chemical differences selected separately for that pair.
Original-report missing values remain marked. No group average is shown.

In the current PDF, chemical names are red and bold if they enter a pair-specific top18
list in at least two distinct distance categories (160 features). The rule is
applied to exact feature IDs and includes features with original-report missing
values. Counts by category are recorded in `tables/pair_pdf_chemical_recurrence.csv`.
Red/bold is only a navigation aid; recurring pairs can share the same sample.
The rule is broad: 160 compounds qualify, covering 26,371 of the 26,604 displayed
name occurrences. Red does not denote a neural association or statistical significance.
The version before red names is retained in `figures/previous_before_red_names/`;
the older version before bold names is in `figures/previous_before_recurrence/`.

`code/build_pair_pdf.py` and `code/pair_pdf_pages.py` build this deliverable with
the Codex bundled Python (ReportLab/numpy/pandas/pypdf). Existing array and pair
tables are reused. The older PNG summary outputs below are retained as working
records, not as the requested deliverable.

Start with [REPORT.md](REPORT.md). Outputs are descriptive inspection resources for Figure 5, not finalized biological or technology-performance claims.

Use the repository's `.pixi/envs/default/bin/python`. All numerical inputs are existing aligned caches. No new dependencies or full Notebook execution are needed.

From an existing Notebook:

```python
from pathlib import Path
import sys

repo = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
out = repo / 'reports/exploration_pair_quadrants_20261002'
sys.path.insert(0, str(out / 'code'))

# Display only, using saved exploration tables.
from plot_quadrants import plot_all
from plot_context import plot_context
plot_all(out)
plot_context(out)

# Only when recomputing this specific exploration is intended:
# from explore_quadrants import run_exploration
# run_exploration(repo, out, tail_fraction=0.25)
```

The latter overwrites this exploration's tables and verification; use another output directory to retain variants. It does not refit the neural representation, redo bootstrap, fit embeddings, or touch source data. For headless plotting set `MPLBACKEND=Agg`.

## Tables and units

- `pair_catalogue.csv`: all 5,565 pairs, four-corner labels, original and alternative neural distances, reference/date/taxonomy context, and existing bootstrap coverage. Chemical units: RMS log2FC difference. Neural units: 1 − cosine.
- `category_summary.csv`, `threshold_sensitivity.csv`: group support and checks at 20%, 25%, 30%. Marginal thresholds are global over all pairs, not per sample.
- `feature_summary.csv`, `class_summary.csv`: all 380 chemical features and normalized report SuperClasses. Relative contribution = feature squared difference / mean squared difference across 380 features, followed by pair averaging. Class values average features, not sums. `mean_distance_share_pct` instead records the additive distance share.
- `selected_features.csv`: 15 named features selected on the same data from the pre-existing complete-report/QC RSD <= 0.30 set; used only for display, never for primary distance definitions.
- `feature_contrasts.csv`, `feature_threshold_sensitivity.csv`, `feature_context_checks.csv`: descriptive within-chemical-band neural-far/near ratios, reference restrictions, absolute differences, complete162-denominator sensitivity, and leave-one-sample ranges. Zero/nonpositive means yield an undefined log ratio (NaN), without a pseudocount; source means are also saved. The complete162 normalization is interpretable as a feature-share baseline only for members of that subset; other rows are merely scaled to that denominator.
- `chemical_distance_quality.csv`: mean per-pair squared-distance fractions linked to original report missingness, QC RSD > 0.30, and the complete162 set. QC and missing masks overlap and must not be added together. Reported means raw `notna()`, not validated detection.
- `sample_pair_degrees.csv`, `reference_stratum_counts.csv`: endpoint reuse and reference overlap. Endpoint-balanced estimates average partners within each participating sample, then samples equally; they do not make the observations independent.
- `neural_cell_summary.csv`: normalized-vector difference decomposition; `0.5*(unit_a-unit_b)^2` sums to 1 − cosine. Not raw amplitude or neuron causation.
- `atlas_pair_order.csv`: exact row IDs for every displayed pair in Figure 03.
- `pair_arrays.npz`: 5,565 × 380 absolute chemical differences, relative chemical contributions, and joint-report masks; 5,565 × 13 neural contributions; feature/cell names and pair IDs. No pickled objects. Row order matches `pair_catalogue.csv`.
- `sample_chemical_log2fc.csv`, `sample_neural_coefficients.csv`, `feature_metadata.csv`: snapshots of inputs needed for interpreting these summaries.

Sources and their SHA256 hashes are recorded in `parameters.json`. Source chemistry is `reports/population_first_20260930/tables/`; source neural data is `reports/exploration_response_profiles_individual_snr_20261002/tables/`. The run reads only those aligned data, not Excel or raw recordings. Only this new output directory is written.

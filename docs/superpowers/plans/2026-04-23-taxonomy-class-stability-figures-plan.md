# Taxonomy Class Stability Figures Plan

> **For agentic workers:** This is a lightweight figure-first plan. Keep the implementation small, script-driven, and easy to inspect. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Starting from the existing taxonomy annotation, generate figures that identify chemical classes with stable association to neural space under fixed-class permutation, repeated reselection, and full-search permutation.

**Architecture:** Add one standalone review script that writes compact tables plus publication-not-ready exploratory figures. Do not extend the archived supervised subspace search route, and do not create a heavy reusable framework unless the results justify it later.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, existing `bacteria_analysis` helpers. No new dependencies.

---

## Scope

Default candidate scope:

- Use taxonomy `Class::<category>` as the main unit.
- Keep `SuperClass` and `SubClass` optional CLI switches or supplementary outputs only if easy.
- Use the current neural reference: filtered `202604_without_20260331`, non-ASE L/R merge + trial median + correlation distance.
- Use chemical features from `data/matrix.xlsx` after `QCRSD <= 0.2`, `log2(matrix)`, and Euclidean distance.

Non-goals:

- No paper-ready wording.
- No metabolite-level greedy search.
- No heavy test suite.
- No polished manuscript panel layout yet.

## Inputs

Primary data:

- `H:/Process_temporary/WJH/bacteria_analysis/data/202604/202604_preprocess_without_20260331`
- `H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx`
- `H:/Process_temporary/WJH/bacteria_analysis/data/metabolism_raw_data.xlsx`

Useful existing context only:

- `H:/Process_temporary/WJH/bacteria_analysis/results/202604_without_20260331/backup/20260422_result_refactor_unused/taxonomy_category_rsa_qc20_log2_euclidean/taxonomy_category_rsa_summary.csv`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202604_without_20260331/backup/20260422_result_refactor_unused/taxonomy_category_rsa_qc20_log2_euclidean_per_date/top_taxonomy_categories_per_date_summary.csv`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202604_without_20260331/backup/20260422_result_refactor_unused/partial_overlap_subspace_fusion_qc20_log2_euclidean/weighted_fusion_summary.csv`

Implementation references:

- `H:/Process_temporary/WJH/bacteria_analysis/scripts/plot_biological_subspace_individual_rdms.py`
- `H:/Process_temporary/WJH/bacteria_analysis/scripts/plot_biological_subspace_rdm_panel.py`
- `H:/Process_temporary/WJH/bacteria_analysis/scripts/plot_neural_chemical_rdm_foundation.py`

## Outputs

Create:

- `H:/Process_temporary/WJH/bacteria_analysis/scripts/taxonomy_class_stability_review.py`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202604_without_20260331/taxonomy_class_stability_review`

Main figures:

- `figures/01_fixed_class_permutation.png`
- `figures/02_reselection_stability.png`
- `figures/03_full_search_permutation.png`
- `figures/04_taxonomy_class_stability_summary.png`

Tables needed for figures:

- `tables/class_observed_scores.csv`
- `tables/fixed_class_permutation_summary.csv`
- `tables/reselection_runs.csv`
- `tables/reselection_stability_summary.csv`
- `tables/full_search_permutation_summary.csv`
- `tables/final_class_shortlist.csv`

## Evidence Layers

Layer 1: fixed-class permutation

- For every eligible class, compute observed RSA against the neural RDM.
- For every class, keep the class fixed and shuffle stimulus labels on the chemical RDM.
- Output class-wise null quantiles, empirical p-value, and observed percentile.
- Interpretation: conditional evidence that this fixed class has above-chance alignment.

Layer 2: reselection stability

- Repeatedly resample stimuli, preferably date-stratified when possible.
- In each resample, rerun class scoring and select top classes from scratch.
- Track how often each class appears in top 1, top 3, and top 5.
- Track whether related classes recur together or replace each other.
- Interpretation: evidence that class selection is not one-sample fragile.

Layer 3: full-search permutation

- For each permutation, shuffle stimulus labels and rerun the complete class search.
- Store the best null RSA from the full search, not only one fixed class.
- Compare observed best class RSA to this best-of-search null.
- Interpretation: search-corrected evidence against selection luck.

## Task 1: Build The Lightweight Script

**Files:**

- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/taxonomy_class_stability_review.py`

- [ ] Parse CLI args for input roots, output root, taxonomy level, seed, fixed permutations, resamples, and search permutations.

Suggested defaults:

```text
--taxonomy-level Class
--fixed-permutations 2000
--resamples 500
--search-permutations 2000
--resample-fraction 0.8
--top-k 5
--seed 20260423
--output-root results/202604_without_20260331/taxonomy_class_stability_review
```

- [ ] Reuse helper logic from `plot_biological_subspace_rdm_panel.py` where practical: stimulus mapping, neural RDM construction, taxonomy QC loading, and chemical RDM construction.

- [ ] Keep implementation local to the script if importing helpers becomes messy. Clarity beats abstraction here.

## Task 2: Compute Fixed-Class Scores And Fixed Permutations

**Files:**

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/taxonomy_class_stability_review.py`

- [ ] Build one chemical RDM per eligible class.

- [ ] Compute observed RSA for `response_window` and optionally `full_trajectory`.

- [ ] For each class, run fixed-class label-shuffle permutations.

- [ ] Write `tables/class_observed_scores.csv` and `tables/fixed_class_permutation_summary.csv`.

- [ ] Render `figures/01_fixed_class_permutation.png`.

Figure design:

- Dot plot ordered by observed `response_window` RSA.
- One row per class.
- Dot = observed RSA.
- Thin interval = fixed-class null q95 to q99.
- Color = empirical p-value or observed percentile.
- Label only the top stable candidates to avoid clutter.

## Task 3: Add Reselection Stability

**Files:**

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/taxonomy_class_stability_review.py`

- [ ] For each resample, sample about 80% of stimuli, preserving dates when feasible.

- [ ] Recompute all class scores inside the resample.

- [ ] Save the top `k` classes per resample.

- [ ] Write `tables/reselection_runs.csv` and `tables/reselection_stability_summary.csv`.

- [ ] Render `figures/02_reselection_stability.png`.

Figure design:

- Left panel: bar plot of top-1 / top-3 / top-5 reselection frequency.
- Right panel: small heatmap of class recurrence by date-resample support or view.
- Sort classes by top-3 frequency, then observed RSA.

## Task 4: Add Full-Search Permutation

**Files:**

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/taxonomy_class_stability_review.py`

- [ ] For each permutation, shuffle stimulus labels once.

- [ ] Rerun the full class search under that shuffle.

- [ ] Store the best null RSA and the null-winning class.

- [ ] Write `tables/full_search_permutation_summary.csv`.

- [ ] Render `figures/03_full_search_permutation.png`.

Figure design:

- Histogram of best-of-search null RSA.
- Vertical line for observed best class RSA.
- Optional second line for the best reselection-stable class.
- Text annotation with empirical search-corrected p-value.

## Task 5: Make The Final Summary Figure

**Files:**

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/taxonomy_class_stability_review.py`

- [ ] Combine the three evidence layers into `tables/final_class_shortlist.csv`.

- [ ] Create a compact scorecard figure: `figures/04_taxonomy_class_stability_summary.png`.

Figure design:

- Rows = shortlisted classes.
- Columns = observed RSA, fixed-class percentile, reselection top-3 frequency, search-corrected status.
- Use simple labels: `fixed signal`, `reselected`, `search-corrected`.
- Avoid strong wording such as `validated` or `paper-ready`.

Shortlist rule for this exploratory pass:

- Keep classes with positive observed RSA.
- Keep classes above their fixed-class null q95.
- Prefer classes with top-3 reselection frequency above 20%.
- Mark search-corrected pass/fail separately; do not hide classes that fail it.

## Task 6: Minimal Verification

No heavy test suite. Use these checks:

- [ ] Run Python compile:

```powershell
python -m py_compile scripts/taxonomy_class_stability_review.py
```

- [ ] Run a smoke pass with tiny counts:

```powershell
pixi run python scripts/taxonomy_class_stability_review.py --fixed-permutations 20 --resamples 20 --search-permutations 20 --output-root results/202604_without_20260331/taxonomy_class_stability_review_smoke
```

- [ ] Run the figure pass:

```powershell
pixi run python scripts/taxonomy_class_stability_review.py --fixed-permutations 2000 --resamples 500 --search-permutations 2000
```

- [ ] Confirm all four main PNGs exist and the summary CSV is non-empty.

## Interpretation Guardrails

- Fixed-class permutation can support a fixed class's conditional signal.
- Reselection stability supports whether that class keeps being found.
- Full-search permutation supports whether the selection procedure beats search luck.
- A class should be called `stable candidate` only if it has evidence in all three layers.
- If the best RSA class and the most stable class differ, report both visually and do not force a single winner.
- Current outputs remain exploratory figures, not paper results.

## Runtime Controls

If runtime is high:

- First reduce `--search-permutations`, not the fixed-class permutation.
- Keep `--resamples` at least 200 for readable stability frequencies.
- Restrict to `--taxonomy-level Class` before adding `SuperClass` or `SubClass`.
- If needed, generate response-window figures first and make full-trajectory a sensitivity rerun.

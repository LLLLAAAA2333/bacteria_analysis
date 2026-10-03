# Direct-report chemical co-change and neural combinations

Date: 2026-10-03. This is the current exploration after the user rejected unexplained legacy chemical denominator groups. It starts from original report values and recomputes chemical eligibility, chemical modules, model selection and correspondence. It does not import old fold changes, reference mappings, chemical feature lists, candidate memberships or fitted associations.

## What is newly computed and what is retained

| Input | Use in this exploration |
| --- | --- |
| `data/metabolism_raw_data.xlsx`, `all` sheet | Read all 380 reported chemical rows directly; align the 106 neural strain IDs to sample columns. |
| Same workbook, `QC-1` to `QC-39` | Recompute each feature's sample SD / mean over observed QC values; retain RSD ≤ 0.30 and at least two observed QC values. |
| Same workbook, `A_vs_B`, `Name` and `unit` fields only | Recover report-unit metadata. These rows say ng/mL. Source audit verified the sample/QC values agree with the `all` sheet after name alignment. No contrast, FC or grouping results from that sheet are used. |
| `data/GM300_bacteria_species_summary.xlsx`, `Axxx_species_mapping` | Read genus/species and source notes directly; use genus for descriptive checks only. |
| Current `strain_coefficients.csv` and `condition_metrics.csv` | Retain the already defined 13-cell, signed, SNR-screened template representation. Verify that date-equal aggregation of condition coefficients reproduces the strain table; calculate L2-normalized strain vectors again. |

The neural templates have not been re-estimated from calcium recordings. The chemical input does not divide by a control/sample/pooled reference and is not centered by old chemical groups. The random split, regression, baselines and figures do not use those groups.

## Fresh chemical panel

The 380 × 106 report region contains 3,962 missing entries. Source missingness is preserved in `tables/chemical_report_all_380.csv` and is not replaced by zero. There are 173 features with finite positive values in all 106 strains. Intersecting this condition with the newly computed QC rule retains **162 features**; the number is an outcome of recomputation, not an imported panel definition. The retained features have 36–39 observed QC values each. Recomputed QC RSD agrees with the report column to about 5e−15.

All retained values are positive, so the chemical transform is

`x = log2(reported concentration / (1 ng/mL))`.

No pseudocount is needed or added. The report unit does not establish the concentration actually delivered during neural recording; chemical material was independently cultured.

## New candidate and observed correspondence

The single selected chemical module contains **Glucaric acid, Lumichrome, and Vitamin B1**. It spans three Mass-column families and two chromatography columns. These are report annotations, not a validated common pathway.

The three chemicals load positively. For each, `z` is its log value minus the discovery mean, divided by the discovery SD. The frozen score is approximately:

`0.36317 × z(Glucaric acid) + 0.36691 × z(Lumichrome) + 0.37400 × z(Vitamin B1)`.

Its discovery SD is one. A larger score means the three report concentrations tend to be higher together relative to the discovery sample distribution. Full coefficients, means and scales are in `frozen_candidate.json`.

| Diagnostic | Result |
| --- | ---: |
| Discovery / holdout strains | 70 / 36 |
| Newly formed eligible chemical modules | 11 |
| Selected module | M11, 3 annotations / 3 Mass-column families |
| Chemical PC1 variance fraction, discovery | 82.02% |
| Median chemical pairwise Spearman correlation, discovery / holdout | 0.747 / 0.876 |
| Minimum chemical pairwise Spearman correlation, holdout | 0.801 |
| Discovery 5-fold CV full-vector improvement over training-mean baseline | +6.45% |
| Chemical score vs fixed neural projection, discovery / holdout | 0.544 / 0.509 |
| Holdout complete-vector prediction improvement over discovery-mean baseline | +2.10% |
| Discovery–holdout complete neural slope cosine | 0.803 |
| Holdout slope along frozen direction / discovery slope | 57.19% |
| Within-genus holdout correlation | 0.244 |
| Within-genus support | 24 strains, 6 genera, 18 residual degrees of freedom |
| Genera with positive within-genus slope | 3 / 6 |

The main shared changes are **ADF, AWB and ASH increasing, and AWCON decreasing**, with AWA increasing and ASK decreasing as well. ADF has similar slope in both splits; AWCON and ASH slopes attenuate in holdout. Several smaller components change sign, including ADL, ASI, ASJ, ASEL, ASER and AWCOFF. Every analysis and figure retains all 13 cells. The coefficients are signed **unit template coefficients**; their increase/decrease is not a direct excitatory/inhibitory label, a firing rate or a calibrated activity total.

The new candidate has partial support for a repeated combination-level association, with a modest quantitative prediction benefit. It is not an exact reproduction of the entire neural profile. Projection correlation and slope cosine are related geometric diagnostics, not independent replications.

## Genus and other descriptive checks

The pooled association is not removed by deleting any one genus from holdout: the correlation ranges from 0.422 to 0.572; omitting Bacteroides gives 0.422. However, within-genus correspondence is uneven. Bacteroides (11 holdout strains) has a near-zero/slightly negative slope; Bifidobacterium (4) and Pediococcus (3) have positive slopes. Lactobacillus, Escherichia and Limosilactobacillus each have only two holdout strains, so their slopes are highly uncertain. The pooled result must not be described as a universal within-genus rule.

The chemical score correlates 0.580 with the selected 162-feature panel's mean log concentration in holdout. Removing this single level descriptor from both scores gives a descriptive residual correlation of 0.473. This is not adjustment for all cultivation/measurement factors or a new predictive validation; mean log concentration is not total chemical concentration.

With the chemical score and split fixed, pre-gate coefficients projected onto the same neural templates give holdout correlation 0.417, neural slope cosine 0.778, and full-vector improvement +0.65%. Only the neural direction is fitted again on discovery strains for this sensitivity; no chemical module is reselected.

## Frozen search rules

1. One **unstratified random** 70/36 strain split, seed 20261003. No old group labels or genus labels affect the split.
2. In discovery chemistry only, form average-linkage clusters from `1 − |Spearman rho|`, cut at 0.30. Retain at least three annotations, three Mass-column families and PC1 variance fraction ≥0.50. These are fixed exploratory simplification rules, not biological boundaries. Average linkage does not require every pairwise |rho| to exceed 0.70.
3. Use discovery-only feature means/SDs. Apply inverse-square-root family-count weights to limit repeated annotation weighting, then fit PC1; orient the largest effective weight positive and normalize discovery score SD to one. Shared Mass-column families are a redundancy audit, not proof of molecular identity.
4. Rank modules using five discovery folds and full 13-vector squared prediction error relative to each fold's training neural mean. Each fold refits means, SDs, PC1 and neural regression; module memberships are fixed from the outer discovery chemistry. The CV score is for candidate selection.
5. Freeze one candidate and its 13-dimensional neural direction before the holdout calculation. Evaluate only that candidate. Use one intercept and one score, with the discovery neural mean as the prediction baseline. No second candidate or alternate split is selected after viewing the result.

This dataset, including earlier heldout strains, has already been explored. The new split is an **internal exploratory train/holdout check on previously seen data**, not a new independent experiment. The representation uses existing atlas templates and the chemical completeness screen uses whole-dataset availability. Shared animals and unknown medium/culture/measurement structure remain possible sources of dependence or association. Removing an unexplained grouping assumption does not establish the absence of those effects.

## Files

- `figures/01_candidate_correspondence.png` / `.svg`: frozen chemical/neural scores in each split and all 13 neural change coefficients.
- `figures/02_holdout_profiles.png` / `.svg`: all 36 holdout strains sorted by chemical score, three chemical features beside all 13 neural coefficients, same rows.
- `figures/captions.txt`: English external figure descriptions.
- `tables/fresh_feature_audit.csv`: every original chemical feature, completeness, recomputed QC and inclusion decision.
- `tables/selected_members.csv`, `neural_slopes.csv`, `holdout_genus_summary.csv`: interpretable feature and neural changes with coverage.
- `code/notebook_cells.py`: cells for reading/displaying results without rerunning selection.
- `code/direct_report_recurrence.py`: small importable functions for source preparation, discovery and frozen evaluation.
- `parameters.json`, `source_manifest.json`, `frozen_candidate.json`, `results.json`: exact settings, original-source hashes, fixed model and outputs.

Previous outputs are retained as historical records but are not inputs to this analysis. Since the chemical transform, split, adjustment and resulting candidate differ, changes in the findings cannot be attributed solely to deleting the old grouping.

# Asparagine: bounded second-stage check and record of refutation

This subanalysis was added after the original 162-feature × two-neural-target screen. Asparagine was selected because it had the largest absolute **full-adjusted ADF** correlation, after seeing the animal-only screening results. It was not a prior hypothesis. The Bacteroides-only contrast, six high-report-strain deletion, species controls, and selection-rule sensitivity are also post hoc. No p values or confirmatory claims are produced. This record must accompany any retained finding.

## Main judgment

The Asparagine report helps describe a Bacteroides **between-species / high-versus-low chemical-profile contrast**, but these data do not isolate an Asparagine-specific response mechanism or a transferable dose-response law. Dropping one strain or date does not resolve this competing explanation; controls aimed at its multistrain and taxonomic structure do.

The neural pattern is concrete: A024/A025 have relatively high ADF stimulus responses and Asparagine reports of 827.722/1002.364 ng/mL, whereas A013/A014 have lower ADF responses and reports of 92831.608/74273.355 ng/mL. Chemical reports have no verified culture/exposure-batch pairing to neural stimuli. Units denote collaborator-reported values, not independently calibrated worm exposure concentrations. Asparagine is detected in all 106 linked strain profiles, with 39 QC injections and QC RSD 0.22632; these are not 39 biological culture replicates or confirmation of structural identity.

## Observation unit and model

All fits use one mean [0,10) s ADF calcium response per strain/date/animal. Shared animals have intercept blocks `(date,worm_key)`; each strain/date has total weight one. No trial is a biological replicate. The whole dataset has 499 ADF animal–strain observations from 40 actually measured ADF animals, 106 strains and nine dates. Bacteroides contributes 163 observations, 22 ADF animals, 29 strains and five dates; the other genera contribute 336 observations, 35 ADF animals, 77 strains and eight dates. Animals can appear in both genus subsets.

`animal` projects out weighted animal intercepts; `full` also projects out chemical reference group, median log chemical profile level, and genus; `species` adds species indicators. Rank-truncated weighted SVD handles exact indicator dependencies, matching the corrected implementation in `05_chemical_tests.py`. Correlations/slopes are descriptive conditional associations. Neither fixed effects nor these deletions disentangle arbitrary strain-by-date effects.

## Results and strongest alternatives

| Scope | Animal r | Full r | Full + species r |
|---|---:|---:|---:|
| All 106 strains | -0.40262 | -0.41943 | +0.10160 |
| Bacteroides, 29 strains | -0.59505 | -0.65178 | +0.03029 |
| Other genera, 77 strains | -0.09592 | +0.12967 | +0.01208 |
| Bacteroides minus A024/A025, 27 strains | -0.49454 | -0.54234 | +0.03993 |
| Bacteroides minus all six cross-date strains, 23 strains | -0.39875 | -0.45890 | +0.02531 |
| Bacteroides minus uniformis + fluxus, 25 strains | -0.48097 | -0.58061 | -0.00253 |
| Bacteroides minus six high-report strains, 23 strains | +0.04995 | -0.16967 | +0.00055 |

The full-adjusted slope is −0.05347 ΔF/F0 per unit of log2(report + 1) over all strains, and −0.06843 within Bacteroides. These are regression slopes, not experimentally identified concentration effects.

Species controls leave only 10.40% of the animal-adjusted Asparagine residual SD overall and 8.53% within Bacteroides (approximately 1.08% and 0.73% of residual variance). There are 67 species among 106 strains, 28 with multiple strains; Bacteroides has 16 species, ten with multiple strains. The near-zero species-adjusted Bacteroides correlation therefore **does not establish absence of a molecular effect**: little within-species chemical support remains. It does refute presenting the original genus-adjusted correlation as distinguished from species identity.

The six high-report Bacteroides strains are A009/A013/A014/A019/A020/A044, identified post hoc from the chemical gap. Removing them narrows the reported Asparagine range from 471.97–107934.12 to 471.97–4114.18 ng/mL, so this check changes support, not just sample count. Within that lower range, the animal-adjusted gradient is absent in these observations. The original association is not a continuous gradient demonstrably shared across the whole Bacteroides chemical range.

## Checks that did not overturn the broad association

- All-strain leave-one-date-out full correlations range −0.5203 to −0.2832; leave-one-strain-out range −0.4725 to −0.3542; leave-one-species-out range −0.4725 to −0.3503. All deleted-strain/species/date results and support counts are retained.
- Bacteroides leave-one-date-out full correlations range −0.6914 to −0.5891 and leave-one-species-out range −0.6878 to −0.6095. Single deletions miss the joint high/low species-structured explanation.
- Replacing the neural mean with trial median, first retained trial, later-trial mean, exclusion of flagged below−1 traces, or an animal-specific linear order adjustment retains negative Bacteroides full correlations: −0.6418, −0.5470, −0.6176, −0.6616 and −0.6354. The first retained trial is not guaranteed to be first exposure.
- Legacy log-fold-change and log2(report+1) give the same full-adjusted correlation once their reference-group offsets are projected out; this algebraic equivalence is explicitly checked to 1e−10, and is not independent chemical replication. Using untransformed report values gives Bacteroides full r=−0.6633.
- Asparagine versus Aspartic acid report ranks correlate 0.65064 across all 106 strains but only 0.23695 within Bacteroides (29 strains). The all-strain co-variation cannot be imported as evidence of a single Bacteroides mechanism. No new molecular route was pursued.

## Selection-rule sensitivity, not Asparagine validation

All six repeated strains were removed globally, leaving 100 strains. On each of nine held-out dates, the training data select the largest absolute full-adjusted correlation among the same 162 complete-QC features. An animal-only slope for that training-selected feature predicts test-animal-centered ADF responses. Test outcomes do not choose a feature or fit a slope; their centering defines a relative-response evaluation, not absolute new-day prediction.

The aggregate relative R² is 0.14993; RMSE is 0.20495 versus null 0.22229 ΔF/F0; only four of nine dates improve. Asparagine is selected in zero of nine folds. The selected names differ across folds and are retained for transparency, not as new findings to pursue. This assesses the second-stage **selection rule** in the existing discovery data, and does not validate Asparagine independently or justify replacing it with another selected molecule.

## Figure and reproducibility

`figures/05_asparagine_refutation.png` and `.pdf` form one four-panel evidence figure. A shows every measured Bacteroides ADF animal point in original neural units against log2 of the chemical report, with open circles for strain/date means and four explicitly named strains. B and C show animal-intercept residuals within and outside Bacteroides using the same axis limits. Date colours are consistent across panels. D contrasts grouped exclusions and adjustments; its dots are descriptive estimates with no uncertainty bars, not a ranking of molecules.

`tables/asn_sensitivity.csv` records all fits, slopes, sample and species coverage, chemical ranges and residual chemical SD. `asn_plot_points.csv` is the figure source; `asn_strain_data.csv` records chemical and phenotype means; `asn_covary.csv` stores chemical co-variation; `asn_selection_rule_cv_folds.csv` and `asn_selection_rule_cv_predictions.csv` permit full reconstruction of the selection sensitivity. Metadata, input hashes and the shared 05 implementation hash are in `logs/06_asparagine_summary.json`.

Run from the project root:

```sh
OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/python reports/exploration_20260929/code/06_asparagine_check.py > reports/exploration_20260929/logs/06_asparagine.log 2>&1
```

The calculation is deterministic and has no random seed. All requested checks and CV ran successfully. Species-projection residual orthogonality, equivalence after reference normalization, 100 distinct held-out strains, lack of training/test strain overlap, nine folds and exclusion of all six repeated strains were checked in executable assertions. After the parent corrected 05's nearly singular projection, the complete analysis was rerun with that SVD implementation; the initial pinv outputs are superseded. The final figure was opened for visual inspection; the date legend was corrected to include all nine dates, and the B/C axes were aligned. No source data or unrelated outputs were changed.

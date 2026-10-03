# Chemical co-change and complete neural response combinations

Date: 2026-10-03. This is a bounded exploratory search requested after the genus/near–far handoff. It does not determine a final Figure 5.

## Finding

One chemically coherent candidate was identified, but its complete neural change direction did not reproduce consistently in the current internal strain holdout.

The selected four report annotations are **Alpha-aminobutyric acid, N-Acetylalanine, N-Formylglycine, and Pterin**. They have four distinct reported Mass × column families. All four load positively on the chemical axis. Their within-reference median pairwise Spearman correlation is 0.737 in discovery and 0.685 in holdout (holdout range 0.575–0.738). This supports repeatable chemical covariation in this dataset. All four are measured on the Amide column; neither a common pathway nor exclusion of broader sample-level/technical variation has been established.

| Mapping diagnostic | Result |
| --- | ---: |
| Discovery / holdout strains | 70 / 36 |
| Chemical-only candidate modules | 9 |
| Selected module | M05, four annotations / four Mass-column families |
| Discovery PC1 variance fraction | 79.29% |
| Discovery CV full-vector improvement over reference means | +1.85% |
| Holdout correlation with frozen neural direction, within reference | 0.227 |
| Within-reference one-sided permutation benchmark | 0.0966 |
| Holdout full-vector improvement over reference means | −0.62% |
| Discovery–holdout 13-dimensional slope cosine | 0.311 |
| Holdout slope along frozen direction / discovery slope | 37.10% |
| References with positive holdout slope | 2 / 4 |
| Holdout within-genus × reference correlation | 0.144 |
| Within-genus × reference support | 21 strains, 7 cells, 7 genera, 14 residual degrees of freedom |
| Within-genus × reference cells with positive slope | 3 / 7 |

The dominant discovery changes included higher ADF and lower ASEL/AWCON/ASH relative template coefficients. In holdout, AWA and ASH decrease more strongly, ADF increases less, and ASEL/AWCON reverse sign. These are changes in signed **unit template coefficients**, not calibrated total activity, firing rates, or excitatory/inhibitory labels. The fixed 13-cell vector is always retained.

The positive projection correlation and positive slope cosine are mathematically related diagnostics, not two independent replications. The full-vector prediction does not improve over the reference-only baseline. A050 and ref12 have positive holdout projection slopes, while A250 and A306 do not. Small per-reference and per-genus counts limit those descriptive comparisons; cells with only two strains do not provide stable slope estimates.

The primary screened neural representation remains the main result. With the chemical axis and split frozen, the pre-gate same-template and independently refitted unfiltered representations yield holdout correlation about 0.275, slope cosine about 0.400, and vector improvement about +0.61%. Their neural directions were fitted again on discovery strains only; chemistry was neither refitted nor reselected. This sensitivity does not establish a robust full neural combination.

## Frozen search

1. Use the existing 106 × 162 complete-report chemical log table and 106 × 13 neural unit coefficients, aligned by strain. No chemical imputation or recording-date gate.
2. Make one reference-stratified 70/36 strain split, seed 20261003. Singleton genera are not automatically excluded from holdout. Coverage is saved in `tables/split_coverage.csv`.
3. In discovery chemistry alone, subtract discovery reference means; form average-linkage clusters using `1 − |Spearman rho|`, cut at 0.30. Retain modules with at least three annotations, three Mass-column families, and chemical PC1 variance fraction at least 0.50. The linkage threshold does not imply that every pair passes |rho| ≥ 0.70.
4. Standardize features by discovery within-reference residual SD. Weight annotations by the inverse square root of their same-family count, then fit module PC1. This is a conservative redundancy weighting, not molecular identity merging. Orient the largest effective loading positive; chemical score discovery SD is one. This standardized co-change axis differs from the earlier unstandardized chemical RMS distance.
5. Use five discovery CV folds to compare each one-axis, reference-intercept model against reference-only mean neural vectors. Refit feature means, scales, PC1 and neural slope in each fold. Module membership is fixed by outer-discovery chemistry, so this CV is a candidate-selection tool, not an independent effect estimate.
6. Freeze the single highest-ranked module, chemical axis and full neural direction in `frozen_candidate.json` before evaluating the 36 holdout strains. Only that candidate is evaluated in holdout. No reranking or alternative split is performed after seeing its result.
7. Primary holdout diagnostic: correlation of the fixed chemical score and projection onto the fixed neural direction, centered within holdout reference. Secondary diagnostics: full-vector prediction using discovery reference intercepts, 13-vector slope direction, reference-specific association, within-genus × reference association, and fixed-chemistry neural sensitivity.

## Interpretation limits

- This is an **internal holdout of the mapping step**, not a new independent experiment or a fully heldout neural representation. The saved neural templates/SNR were derived from the existing atlas, the complete162 panel used prior whole-dataset availability/QC, and these strains have appeared in previous exploration.
- Strains may share recording animals. The 9,999 within-reference label permutations are an exploratory conditional benchmark and do not model all dependence from shared animals, dates or experimental structure. Do not promote the benchmark to an independent experimental significance claim.
- Reference groups are denominator labels with unresolved experimental identities. Reference centering does not establish cross-reference calibration or remove every potential confound.
- Chemistry came from independent culture material and is not measured stimulus-aliquot concentration. Report annotations do not include supplied identification-confidence levels. No driver molecule, pathway mechanism or behavioral function is established.
- A weak heldout result for this frozen candidate does not prove that every possible nonlinear or context-dependent chemical–neural relationship is absent.

## Figures and reuse

- `figures/01_candidate_validation.png` / `.svg`: discovery and holdout association, plus all 13 neural change coefficients.
- `figures/02_holdout_profiles.png` / `.svg`: all 36 holdout strains, fixed four-feature chemistry and full neural profiles in the same row order. The full chemical color range preserves the extreme A118 value; it is not clipped.
- `figures/captions.txt`: English external captions and display definitions.
- `code/notebook_cells.py`: small cells for reading tables and displaying saved figures in the existing notebook; no automatic rerun.
- `code/chemical_recurrence.py`: importable discovery and holdout functions. They refuse to overwrite frozen results. Parameters, source hashes, membership, split and intermediate tables are saved.

After visual inspection revealed the large A118 chemical value, a **post hoc influence diagnostic** was added in `tables/posthoc_single_strain_influence.csv`. It removes each holdout strain one at a time, recenters within reference and evaluates the already frozen projection. This diagnostic does not alter sample inclusion, the candidate, model fitting or the primary result.

## Verification

Synthetic checks covered reference centering, score SD, signed multivariate slope recovery, prediction and degenerate-permutation handling. A separate read-only calculation starting from chemical logs and unnormalized neural coefficients reproduced the frozen axis, neural direction, heldout correlation, both prediction SSEs, slope cosine and genus-reference result. Main input hashes matched. Both figures were visually inspected.

All outputs are confined to this new report directory. Existing notebooks, data and earlier reports were not overwritten.

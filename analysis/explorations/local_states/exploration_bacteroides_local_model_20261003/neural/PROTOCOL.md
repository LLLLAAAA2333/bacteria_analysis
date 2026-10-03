# Local Bacteroides neural direction: protocol before calculation

Date: 2026-10-03. The user authorized identifying the main neural variation within the 29 recorded Bacteroides strains, followed by a separate chemical explanation. This directory addresses the neural direction only; no chemical measurements, scores, chemical selection, or cross-modal fit are read or performed here.

## Fixed cohort and representation

- Use `reports/exploration_chemical_pattern_direct_report_20261003/tables/neural_unit_coefficients.csv`, its same-template `neural_pre_gate_unit_coefficients.csv`, and `sample_context.csv`.
- Select genus exactly `Bacteroides`; retain all 29 strain IDs and all original 13 neuron columns, sorted by strain ID before calculation. Preserve every species label, `taxonomy_note` and complete semicolon-separated record-date set. Flagged taxonomy rows are kept.
- The saved `strain_coefficients.csv` and matched-context `coefficient_norm` may be read to carry forward the existing amplitude descriptor and check consistency. No raw response template is refitted.
- Unit vectors encode signed template-coordinate composition and remove overall L2 gain. Coordinates and signs do not mean proportions, direct firing, excitation, or inhibition.

## Main decomposition fixed independently of chemistry

1. Subtract the within-Bacteroides mean from each neuron coordinate. Do not standardize by coordinate SD and do not re-normalize the centered strains.
2. Apply one SVD to the 29 × 13 centered matrix. Save all 13 loading vectors, all strain scores, singular values, sample variance `s²/(n−1)`, explained variance ratio, and cumulative ratio. The first PC is the sole prespecified main direction; neither its choice nor orientation uses chemical data. PC1 is a maximal-variance descriptive direction, not a sensitivity estimate or a prediction.
3. Orient each PC so its largest-absolute loading coordinate is positive; in an exact tie, use the original neuron order. This fixes display sign only. Save PC1 reconstructed unit coordinates `mean + score_PC1 × loading_PC1`, residuals, and strain residual L2/RMS values. The reconstruction is not forced back to unit length.
4. Preserve original gated coefficient L2 norm as a separate descriptive magnitude; it does not replace unit coordinates or redefine the primary PC. No selection or exclusion uses this norm.

## Fixed internal stability checks

5. Leave one strain out, once for each of the 29 strains. Refit the neural mean and PCA on the remaining 28 strains. Record absolute PC1 loading cosine to the full-data PC1, training PC1/PC2 variance shares and their gap, aligned 13-coordinate loading, and held-out projection under the training mean/loading. The full-data axis is used only to align signs and summarize direction stability.
6. Leave one recorded species label out, once for each of the 16 labels. Refit using the other strains and save the same direction/variance quantities and held-out projections. Current species labels (including flagged ones) define this descriptive stress test. It does not verify species identities or constitute independent experimental validation.
7. Fit the identical PCA once on the 29 same-template pre-gate unit vectors, using their own mean. Align its PC1 to the primary PC1 only for comparison. Save loading cosine, aligned score Pearson correlation, both spectra, both PC1 loadings, and all strain score differences. No gate search or template refitting.
8. Dates and species are annotation/diagnostic fields, not PCA predictors, filters, or selected parameters. Same animals, dates and globally estimated templates limit independence. Explained variance is within-Bacteroides unit-profile variance; it is not chemical explanatory power.

## Figures and reproducibility

- Two English figures: all-13-coordinate PC1 loading plus complete PCA variance spectrum; all-29-strain neural heatmap ordered only by PC1 score, with species codes, full date sets and taxonomy flags. Main heatmap uses a common symmetric color scale of deviations from the Bacteroides mean, without row or column z-scores. Species-code mapping and full metadata are saved.
- Freeze source hashes, dimensions, parameters and order. Save full matrices and diagnostics, a concise README and figure captions.
- Small callable Python functions support an existing Notebook; no new Notebook or dependency is created. The default display reads saved results. Scientific recomputation requires a fresh output directory and refuses to overwrite existing result tables/core metadata.
- Independently verify the PCA using covariance eigendecomposition and direct projection/reconstruction identities, check stability refits and source hashes, and view both generated figures.

## Display-only refinement after calculation

At the investigator-facing review stage, show both already-computed PC1 and PC2 loadings together because their variance shares are nearly equal, and replace species codes in the heatmap with directly readable B. species labels. This changes only figures/captions; the primary PC1 target, matrices, scores, sample order, stability checks and numerical results remain frozen. The initial figures are archived in `audit/initial_pc1_only_code_labels/`.

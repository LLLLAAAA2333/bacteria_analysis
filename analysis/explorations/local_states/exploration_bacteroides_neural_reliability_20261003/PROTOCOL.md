# Bacteroides neural geometry and repeat reliability

Authorized scope: two parallel neural checks in the existing 29-strain Bacteroides cohort. No chemical model, chemical candidate selection, new notebook or modification of raw/previous analysis outputs.

## Geometry branch

Fix K=2 because the previous neural-only analysis found PC1/PC2 variance ratios 33.60%/31.24%. Use the same 29 × 13 SNR-gated unit-coefficient matrix, neuron centering without column SD scaling, and the same recorded species/date/flag metadata. Describe the complete PC1–PC2 positions.

Refit centered PCA after each of 29 strain deletions and 16 recorded-species deletions. Compare the rank-2 subspace with the full-data reference using both principal angles, their maximum, and normalized projector distance `||P_train − P_full||_F / sqrt(2)`. Report training variance fractions and the gap between eigenvalues 2 and 3. Retain PC1-only cosine as a diagnostic. The full reference overlaps training data: these are deletion-sensitivity checks, not independent replication. Check the same full-data geometry using the saved pre-gate representation. Do not choose dimensionality or thresholds based on chemical associations.

## Repeat branch

Use the previous shared 13-neuron temporal templates as frozen measurement coordinates. This isolates reproducibility conditional on that saved representation; templates were learned from the broader dataset including these animals, so the result is not independent end-to-end validation. Main targets are the fixed two-dimensional projection, unit-profile ADF−ASH, and unit-profile AWB coordinate; also retain the complete 13D difference and original coefficient/norm descriptions. The contrasts are selected now from already observed neural structure, not retrospectively claimed as hypotheses specified before the earlier analysis.

### Recorded-date pairs

Keep all six strains with two recorded dates. Project each complete date-specific 13D profile into the fixed full29 mean/loadings; preserve both dates, recorded species, flags, neuron counts and gate states. Compare the main gated representation with raw coefficients projected onto the same template before gate zeroing. Display individual paired changes, their RMS and a clearly labeled between-strain reference among the same six strains. These six pairs are not 12 independent strains and do not establish independent culture replication; dates and animal records overlap. Never infer a global date correction from this small subset.

### Animal splits

Reuse the 100 saved date-balanced whole-animal assignments; the same animal stays in the same half across all its strain/neuron records. Trials remain averaged within animal. Recompute condition means and the existing SNR formula/gate separately in each half, with the existing minimum of two animals; project onto the frozen shared templates. Suppression is a measured zero; insufficient animal support is missing.

For each strain × date × neuron, use dates with sufficient observation support in BOTH halves, then aggregate those common dates equally within each strain/neuron. Unit normalization and 13D/2D targets require all 13 observed coordinates and nonzero norms in both halves. Do not fill missing neurons with zero. Report coverage and exclusions for every split and strain; subset availability limits generalization to all 29 strains. Gate thresholds, support rules and targets are not tuned to reliability results.

For each split and target on the same jointly supported strain set, calculate:

- `same-strain RMS = sqrt(mean_i ||target_A_i − target_B_i||²)`.
- `different-strain RMS = sqrt(mean_{i != j} ||target_A_i − target_B_j||²)` using all ordered cross-half different-strain pairs, and their ratio.
- For scalar targets, Pearson/Spearman across strain identities as secondary descriptions, without z-scoring the halves separately.
- An additional different-strain reference restricted to identical complete original recorded-date-set labels, with pair counts and NA when unsupported. This is a diagnostic, not a sample filter or batch correction.
- Refit top2 PCA separately in each half on the same fully supported strains and compare the two subspaces. Retain support counts and distinguish this from deletion sensitivity and from fixed-coordinate reproducibility.

Keep full per-split results, not just favorable examples. The 100 splits reuse the same animals; summaries describe variation under these partitions, not 100 independent experiments or inferential confidence intervals. Half-sample RMS is not converted into a full-sample noise ceiling or causal biological variance. Whole-dataset between-strain differences can also contain date/species structure.

## Decision

Assess separately whether a two-dimensional description is sensitive to sample composition and whether its strain positions/contrasts reproduce in animal and date repeats. A stable plane alone does not prove stable strain differences. Report failures or limited coverage directly. End this round with the evidence for or against proceeding to a small local chemical model; do not fit that model in this round.

Each branch preserves source hashes, parameters, complete supporting tables, notebook-callable small functions, output guards and independent numerical checks. Figure text is English; reports explain results in Chinese. Root integration reads saved branch results only.

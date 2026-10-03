# Bacteroides neural top-two subspace sensitivity

Fixed before computation, 2026-10-03. Only the neural subspace branch is covered. No chemical values, chemical groups, prediction models, animal split outcomes or selection of K are read. K=2 is explicitly requested and fixed.

## Sources and representation

Use only the previous neural branch's `neural_unit_profiles.csv`, `strain_metadata.csv`, `pre_gate_unit_profiles.csv` under `reports/exploration_bacteroides_local_model_20261003/neural/tables`. Align by strain ID and the original 13 neuron columns, retain all 29 recorded Bacteroides strains including the six source taxonomy flags, and preserve species and complete date-sets. These rows are previously computed unit template-coordinate profiles: they remove overall L2 gain, and their signs are not a direct excitation/inhibition measure.

Fit centered PCA by SVD to the 29 x 13 gated unit-profile matrix. Subtract each training set's mean across strains; do not divide neuron coordinates by their SD and do not renormalize the centered rows. Sample eigenvalues are s^2/(n_train-1). All 13 components are saved. Orient each component so the largest-absolute loading is positive; ties follow source neuron order, exactly as the prior neural branch. Display sign does not affect the subspace metrics.

## Fixed comparisons

The full29 gated top-two loading span is a descriptive reference. Separately fit all 29 leave-one-strain-out subsets and all 16 leave-one-recorded-species-out subsets. Each fit recomputes its own training mean and complete PCA; record every train/omitted ID. Recorded species define stress-test groups, not independently verified taxonomy. No fold or strain is removed using results.

For each training fit compare its orthonormal top-two loading matrix U (13 x 2) against full29 reference V. The singular values of U^T V are clipped only for numerical roundoff to [0,1]. Ascending principal angles theta1<=theta2 are arccos of descending singular values, in degrees. Save both angles, max angle, and D=||UU^T-VV^T||_F/sqrt(2)=sqrt(sin(theta1)^2+sin(theta2)^2). D ranges from 0 to sqrt(2); it is not an angle or fraction. These metrics are invariant to signs and rotations of the two basis vectors.

Also retain absolute PC1 loading cosine |u1^T v1| and its acute angle, distinguishing stability of a single direction from stability of the plane. Save all training eigenvalues, EVRs, top-two cumulative EVR, absolute lambda2-lambda3 gap, relative gap (lambda2-lambda3)/lambda2 and ratio lambda2/lambda3, plus lambda1-lambda2 for interpreting near-equal first components. No arbitrary stable/unstable cutoff is introduced.

Fit the identical PCA once to all 29 pre-gate unit profiles using their own mean, and compare that top-two span to full29 gated with the same angle/projector/PC1 metrics. No gate thresholds or templates are refit or searched. This is a representation sensitivity check, not an independent replicate comparison.

## Saved outputs and displays

Export full and pre-gate means/loadings/spectra/scores, all 45 fold means/loadings/spectra, and projections of all29 strains using each fold's own training mean/loading with train/omitted status. Preserve complete source matrices/context, exact IDs, source SHA256 hashes, calculation and plotting parameters, source/code versions, and output manifest.

Figure 01 shows all29 PC1-PC2 points twice, using recorded-species and complete-date-set styles. Both plots have identical coordinates, directly readable strain IDs, asterisks for source taxonomy flags, actual B. species names and complete 2026 date combinations. PCA axes are projections in unit-profile coordinate space, not SD units. No cluster labels or classification boundaries are inferred.

Figure 02 shows every max-principal-angle result separately by deletion scheme, plus a comparison to single-PC1 angular change. Highlight the largest plane angle in each scheme by the actual omitted ID/species; save the extrema and all other folds. Display-only point offsets/jitter must not alter metric values. No threshold is used and no selected example replaces the full distributions.

## Interpretation and verification

Full29 contains the training subsets being compared. These analyses characterize sensitivity to deletion/representation and do not constitute independent replication, external validation, or validation of a chemical relationship. Similar two-dimensional spans can coexist with unstable individual PCs because rotation within the plane is allowed. Eigenvalue separation from PC3 is relevant to plane separation; it does not prove biological distinctness. Species/date structure and the same globally estimated templates limit independence. Animal split-half reliability is a separate branch and is not used to choose this result.

Independently verify SVD with covariance eigendecomposition, principal angles with SciPy subspace_angles, projector distance with explicit projection matrices, the trigonometric identity, sign/rotation invariance, all45 fold IDs/metrics, pre-gate comparison, and input/code hashes. Inspect PNGs visually. Small callable functions support existing Notebooks; no new Notebook or dependency. Recalculation requires a fresh output directory and refuses existing core results.

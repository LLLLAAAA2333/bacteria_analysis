# Step 4: independent descriptive analysis of genus-associated neural combinations

Protocol recorded before computation, 2026-10-03. This analysis addresses step 4 only. No chemical matrix, score, feature module, chemical order, or chemical result is read. It does not compare chemical and neural structures (step 5), predict outcomes, or refit response templates.

## Fixed inputs and cohort

- Gated unit neural coefficients, same-template pre-gate unit coefficients, and sample context from `reports/exploration_chemical_pattern_direct_report_20261003/tables/`.
- Optional existing neural `strain_audit.csv` is used solely to describe gate suppression / available replicate support, not to select observations.
- Intersect aligned observed strains, require all 13 neural coordinates, keep every genus with at least two observed strains. Expected: 90 strains, 13 genera. Record the 16 singleton genera separately without using them to select patterns or define the reference.
- Unit vectors remove overall gain. Coordinates are signed coefficients relative to existing neuron-specific templates; positive/negative values do not mean excitation/inhibition and are not percentages. The existing SNR gate can set coordinates to zero; zero is not evidence of no biological response.

## Estimands and ordering

1. For each genus, compute the ordinary mean of its strain unit vectors, without renormalization. Define the reference as the unweighted mean of these 13 genus means. A centered genus profile is its mean minus this equal-genus reference. Each genus therefore has equal weight in the reference, despite unequal strain numbers.
2. Main display includes all 13 genus centers and all 13 neuron coordinates. Genus ordering uses average-linkage Euclidean distance between neural genus centers with optimal leaf ordering. Neuron order is the original input column order. Ordering is solely a display aid, not a tested clustering claim.
3. The 90-strain supporting heatmap subtracts the same fixed reference from every strain, keeps the same genus order, and orders strains by within-genus neural Euclidean distance (average linkage / optimal leaf order; strain ID order when n=2). Do not row z-score either display and do not renormalize mean vectors.
4. Record each genus's center norm (within-genus directional concentration), centered-center norm (departure from the equal-genus reference), per-neuron mean, median, SD, 10th/25th/75th/90th quantiles, and fraction of strains whose centered coefficient has the same sign as the centered genus mean. Fractions are descriptive, not confidence or inference. Full coordinate tables precede textual top-coordinate examples; examples use the three largest absolute centered coordinates, with no selection based on other modalities.

## One fixed stability and sensitivity analysis

5. Leave each strain out once. Recompute its own genus mean and the 13-genus equal-weight reference; other genus centers stay fixed. Save cosine between the remaining-strain centered genus profile and the full centered genus profile, L2 change, and per-coordinate sign retention. Also save cosine of the held-out strain (centered with that leave-one-out reference) to the remaining-strain centered genus profile. This is descriptive held-out alignment of a multineuron direction, not independent replication or predictive model validation. Count held-out cosines >0. No threshold search. Cosine for a zero vector is undefined and saved as missing, not zero. Small n, especially n=2, offers little stability evidence.
6. Recompute all genus profiles with existing same-template pre-gate unit coefficients using its own equal-genus reference. Preserve primary order and coordinate definitions. Report centered profile cosine to primary, L2 change, individual centered-profile cosine, and full per-coordinate differences. This tests only the existing gate choice and does not test template-fitting uncertainty.
7. Describe taxonomy, dates, observed neuron support / gated fraction when available. Dates/strains are not newly reweighted. Species, culture conditions, and date confounding remain uncontrolled; no causal, pathway, behavioral, or chemical-correspondence interpretation.

## Outputs and verification

- Main centered-genus heatmap, supporting all-strain heatmap, whole-combination held-out alignment plot, and same-template pre-gate heatmap.
- Complete membership, reference, raw and centered means, cell summaries, pairwise genus contrast vectors, holdout records, gate sensitivity, metadata/QC, and singleton coverage tables.
- Source SHA-256 manifest, exact analysis/plot parameters and order, reusable small Python functions plus Notebook-call example (no new Notebook), a standalone independent numerical verification, and visual review.
- These are descriptive exploratory patterns in the observed strains. No multiple-comparison p-values, clustering-based biological categories, universal genus claims, or mechanism claims are added.

Cosine guard recorded before calculation: vectors with norm <= 1e-12 have undefined cosine. Centered profiles with norm < 0.05 (5% of a unit vector length; a descriptive guard, not a biological cutoff) are marked `small_offset_direction_caution`; a large cosine alone is not interpreted as stable separation for them. Every directional metric is accompanied by offset norm and within-genus RMS dispersion. This guard does not drop or reorder genera.

Clarifications before interpretation: the reference includes the focal genus (one of 13 equal weights); LOSO recomputes that focal contribution. Unnormalized genus means retain concentration in their vector length, so their Euclidean distances are mean-profile differences rather than pure direction distances. Coordinate sign counts and whole-vector cosine alignment use different estimands. All current vectors have defined cosine. For coordinate signs, sign(0)=0 and only equal signs count; a zero sign is not a direction. The held-out positive fraction uses the number of defined held-out cosines as its denominator.

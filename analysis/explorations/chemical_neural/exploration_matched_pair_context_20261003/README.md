# Strict context-matched neural–chemical inspection

The user authorized exploration of when neural response combinations remain
similar or change at comparable chemical distances, and explicitly selected
strict matching of reference, neural recording dates, and genus. This directory
contains that bounded exploration. It does not fit a new chemical–neural model
or select individual molecules as neural drivers.

The output PDF is `output/pdf/neural_chemical_matched_context.pdf` relative to
the repository root. It contains a guide, an inspection map of all 100 eligible
pairs, the single unmatched neural-far pair, and every one of the 121 qualifying
anchor comparisons. Case-to-page mapping is in `tables/matched_cases.csv`.

## What the strict comparison supports

- Same reference plus identical full recording-date support permits 343 pairs.
  Adding same genus leaves 100 pairs: 82 neural-near, 17 middle, and 1 neural-far
  under the existing global 25% neural tails.
- There are **no** strictly matched near-versus-far controls with chemical
  distances within 10%. The result also holds at 20%/30% neural tails and
  5%/15% chemical calipers, including a search that does not require a shared
  anchor. This is a limitation of comparison support, not evidence that
  discordant chemical/neural relationships do not exist.
- The sole far pair is A291/A296 (Pediococcus; A306 reference; 20260520): chemical
  distance 0.9643, neural distance 0.8795. It is chemical-near under the new
  complete162 definition. No third same-genus strain has the same reference and
  date support. It is shown separately rather than treated as a matched control.
- Without requiring opposite neural tails, 121 anchor comparisons satisfy the
  strict background and 10% chemical-distance match: 25 anchors, 31 strains,
  six reference/date/genus contexts. Every case is plotted, in context and ID
  order. Partner B has the lower neural distance to the anchor and C the higher;
  these names do not relabel a middle-distance sample as far.
- The 121 cases comprise 85 near/near, 28 near/middle, and 8 middle/middle arm
  combinations. Their neural-distance gaps range from 0.00093 to 0.30228, with
  median 0.06460. The gaps are nonnegative by ordering, and the cases reuse
  strains; these are not independent replication counts or an occurrence-rate
  estimate for a biological population.
- Coverage is uneven: 112 of the 121 cases are Bacteroides, including 79 from
  the single A050 / 20260601 context. Six contexts do not imply six equally
  supported or independent replications.

## Individual cases worth inspecting

These examples illustrate the range in the full inspection PDF. They are not
independent validation or final Figure 5 selections.

| Case / PDF page | Anchor; partners B, C | Chemical distances | Neural distances | Unfiltered neural-distance gap |
| --- | --- | --- | --- | --- |
| M119 / 122 | A041; A049, A045 | 0.9708, 0.9797 | 0.0480, 0.3372 | 0.2782 |
| M082 / 85 | A021; A016, A015 | 1.6536, 1.6091 | 0.0799, 0.3822 | 0.1222 |
| M114 / 117 | A189; A179, A178 | 1.1257, 1.0534 | 0.2522, 0.2593 | 0.0038 |

M119 has chemical distances within 0.92%, yet one neural distance is near and
the other middle. The unit response moves from an ADF-dominated profile toward
a larger relative ASH coefficient, with additional differences across the
other cells. This is not just common amplitude scaling: the anchor and C norms
are 0.8019 and 0.8425. The primary distance gap is 0.2891 and remains 0.2782 in
the saved independently fitted unfiltered representation. However, A041/A049
are annotated Bacteroides kribbi and A045 Bacteroides massiliensis: genus is
matched, species is not.

M082 is the largest primary distance gap among the 121 displayed cases, but it
shrinks from 0.3023 to 0.1222 without SNR screening. It therefore illustrates
why a striking screened profile should not be treated as a filter-independent
effect. All three strains have different species annotations within Bacteroides.

M114 illustrates nearly equal neural distances at comparable chemical
distances. Neither pair is in the global neural-far tail. Such comparisons are
kept rather than displaying only the cases with large gaps.

The separate A291/A296 pair has coefficient norms 0.5268 and 2.8782. The second
profile has a strong AWCON coefficient; both scale and direction differ.
Its unfiltered neural distance remains 0.8201. The pair is two Pediococcus
species and cannot supply the missing within-context matched control.

## Definitions

**Chemical coordinates.** The fixed pre-existing 162-feature panel requires
an original report value in all 106 strains and QC RSD <= 0.30. Every value is
finite and positive. Coordinates are `log2(original reported value + 1)`;
distance is the RMS coordinate difference. No missing value is imputed here,
and the report values are not reinterpreted as concentrations delivered to worms.
The pseudocount is fixed in original report units, as in the existing workflow.

Within a common reference, the original fold-change denominator cancels in
pairwise log differences. On this fixed panel the computed distances equal
the corresponding existing log2FC distances to 2.45e-15. Thus sample- or
reference-side zero filling in the old 380-feature distance does not enter
these within-reference complete162 pair differences. This does not fix unknown
experimental reference identities or recover unreported compounds.

Chemical near/middle/far labels use global 25%/75% thresholds across all 5,565
complete162 distances: 1.2034716540 and 1.6467429244. These are new descriptive
labels for the 162-feature reported-value geometry, not the old 380-feature
labels. They do not establish chemical equivalence, and cross-reference pairs
are excluded from the strict comparisons.

**Neural coordinates.** The saved 106-by-13 signed coefficients use the existing
individual SNR >= 0.5 templates. Raw coefficients retain delta F/F0 units; unit
coefficients divide each strain vector by its L2 norm. Neural distance remains
1 minus cosine. Global near/far limits are 0.2825899144 and 0.7003262069.
Screened zeros do not establish physiological absence. All 106 vectors are
finite and nonzero.

**Background.** All three strains have identical numerical chemical reference,
the same entire recording-date support set, and the same genus. Each cell has
the same equal-date aggregation support within each strain, verified from
condition-level records. This does not control species, culture state, stimulus
order, or all animal effects. Only 3 of the 121 cases have all three species
annotations identical. Identical date support is not a randomized experimental
control or proof of independent animal sampling.

**Chemical distance matching.** An anchor A is compared with different partners
B and C. Require

`2 * abs(d(A,B) - d(A,C)) / (d(A,B) + d(A,C)) <= 0.10`,

with a positive denominator. The two distances' lengths are matched, not their
chemical difference vectors. Across the main cases the cosine of those two
chemical change vectors ranges from -0.0227 to 0.9772. Signed changes for all
162 features are retained. The 18 displayed names are selected symmetrically
by `sqrt((delta_B**2 + delta_C**2)/2)` and shared by the two arms; this is a
display rule, not molecular association screening.

**Amplitude and direction.** The raw/unit profile panels and vector norms are
both retained. The numerical audit uses

`||a-b||^2 = (||a||-||b||)^2 + 2*||a||*||b||*(1-cosine)`.

This separates the Euclidean coefficient difference into norm and direction
terms without interpreting the norm as calibrated total neural activity.
Similarly, chemical RMS squared splits into squared mean log shift plus
centered log-change RMS squared. Mean log shift is not total molecular abundance.

## Sensitivity and inference limits

At chemical calipers 5%, 10%, and 15%, there are 64, 121, and 159 strict
continuous-distance comparisons. None spans the two global neural tails at
20%, 25%, or 30%. These settings were declared as sensitivity checks, not tuned
to recover an extreme contrast.

The lower-to-higher neural ordering is preserved without filtering in 108 of
121 cases. This reuses the same strains and is a descriptive sensitivity check,
not a validation rate. Existing pre-gate, SNR 0.25/0.75, raw-curve distances, and
bootstrap valid-draw fractions are also preserved per arm. The valid-draw
fraction is not a confidence level or the probability a distance is correct.

All cases share samples; many also share the same measured animals and stimulus
catalogue. No ordinary pairwise p-values, equivalence tests, causal claims, or
technology-performance claims are made. Chemical measurements are reference
profiles from separate cultures, not measured aliquots delivered in the neural
experiment. Missing mechanisms, annotation certainty, and calibration remain
unresolved outside the complete162 inspection scope.

The narrower coverage is the main substantive result: the original four-corner
question remains useful, but the present strict dataset does not supply a
balanced set of near-versus-far matched controls. The PDF exposes the available
continuous changes and the unmatched far example without relaxing the user's
background criteria.

## Files and reproduction

- `tables/all_pair_catalogue.csv`: all 5,565 pairs, both old/new chemical distances,
  exact-context flags, continuous neural/chemical measures, and sensitivity values.
- `tables/strict_context_pairs.csv`, `strict_context_coverage.csv`: 100 eligible
  pairs and the complete strict-context coverage, including singleton strata.
- `tables/matched_cases.csv`: the 121 plotted cases and PDF page index.
- `tables/case_chemical_changes.csv`: all 121 × 162 signed chemical changes.
- `tables/unmatched_far_pairs.csv`: the sole strict-context neural-far pair.
- `tables/matching_sensitivity.csv`, `strict_opposite_tail_pair_controls.csv`:
  predefined caliper/tail checks. The latter does not require a shared anchor.
- `tables/candidate_matches_up_to_15pct.csv`, `excluded_cross_genus_anchor_cases.csv`:
  audit records of candidates excluded from the strict main PDF. Cross-genus
  anchors are not presented as matched primary evidence.
- `tables/sample_context.csv`, `neural_coefficients.csv`,
  `neural_unit_coefficients.csv`, `chemical_log2_report_plus1.csv`,
  `feature_metadata.csv`: aligned plotting inputs with units defined above.
- `parameters.json`, `observations.json`, `verification.json`: exact settings,
  descriptive counts, numerical invariants, and protected source hashes.

`code/explore_matched_context.py` exposes `run_exploration(repo, out, caliper=.10,
tail=.25)` for use from an existing notebook with the repository's
`.pixi/envs/default/bin/python`. Calling it overwrites this exploration's tables;
use a different output directory to preserve variants. No original notebooks,
raw data, templates, embeddings, or bootstrap samples are modified.

`code/plot_matched_context.py` reads the saved tables and regenerates the PDF
and PNG/SVG previews. `figures/plot_verification.json` records page/order,
plotted-value, text, and rendered-layout checks for the final PDF.

An independent read-only review reconstructed the distances, enumerated the
121 cases, checked dates/genus and every signed chemical row/top18 rank, and
verified the norm decomposition and unchanged input/notebook hashes.

# Exploratory neural–compound associations

This is the requested investigation of possible relationships, using existing
strain-level data. It is an inspection resource for deciding what Figure 5
could show. The 13-page PDF is `output/pdf/neural_compound_relation_candidates.pdf`
relative to the repository root: a reading guide and one page for each of twelve
compound–cell pairs. Every relationship page retains all 106 strains, showing
chemical log2FC versus the signed neural coefficient, adjusted rank residuals,
and the four chemical reference groups separately. The first six relationships
also have PNG and SVG copies under `figures/`.

## Observations to inspect

Values below are descriptive partial rank correlations after adjustment for
chemical reference and neural recording date. They are not effect sizes in
fluorescence units, independent validation, or multiplicity-adjusted discoveries.

| Compound and cell | Coefficient rho | Unit coefficient rho | Coefficient rho with genus adjustment | PDF page |
| --- | ---: | ---: | ---: | ---: |
| Lactate–AWA | -0.382 | -0.366 | -0.332 | 2 |
| Ureidopropionic acid–AWA | +0.357 | +0.365 | +0.292 | 3 |
| 2-Hydroxy-3-methylbutyric acid–ASEL | +0.305 | +0.299 | +0.285 | 5 |

- Lactate–AWA has a negative direction in all four reference groups. Removing
  one entire reference group and refitting gives rho from -0.422 to -0.292;
  the separately fitted unfiltered representation gives -0.386. These checks
  are consistent with a candidate association, not a compound-specific effect.
- 2-Hydroxy-3-methylbutyric acid–ASEL has a positive direction in all four
  reference groups. Removing one reference gives +0.243 to +0.367; the
  unfiltered representation gives +0.258. The raw scatter is still dispersed,
  with many small or screened-zero coefficients.
- Ureidopropionic acid–AWA has a positive overall adjusted association, but
  A250 has rho -0.155 (15 strains), whereas the other three references are
  positive. It is less consistent across reference groups.

The full twelve-pair selection and all screening results are saved, including
weaker or inconsistent subgroup patterns. Cross-compound covariance limits
specificity: Lactate also correlates with Phenylalanine (adjusted rho +0.482),
Leucine (+0.470), and other chemical features. A single-compound intervention
has not been measured here. The chemical measurements come from separate
cultures, not the actual aliquots used for the neural recordings.

The old Arginine–AWCON example is not supported as a consistent association by
this screen: adjusted coefficient rho is -0.142, with mixed reference-group
directions. A compound appearing frequently in the pair atlas is not by itself
evidence of an association with a neuron.

## Primary analysis

- Unit of analysis: 106 strains, with 13 neural cells. The 5,565 strain pairs
  are not treated as independent observations.
- Chemical inputs: the existing log2FC relative to the assigned numerical
  reference. These groups identify denominator source columns; their experimental
  identities as media, blanks, or batches have not been established by the
  available metadata. The primary set consists of the pre-existing 162 features with
  original report values for every strain and report QC RSD <= 0.30. No absent
  chemical report value is imputed in this investigation.
- Neural inputs: the existing SNR >= 0.5 signed template coefficients, in
  delta F/F0 because each template has unit RMS. Screened zeros do not establish
  absence of a biological response. A coefficient's sign is relative to its
  cell's template; a negative association does not directly establish inhibition.
- Unit coefficients divide each strain's entire 13-cell coefficient vector by
  its L2 norm, `sqrt(sum(coefficient**2))`. This removes a common positive
  amplitude scale. It does not make coefficients sum to one or make them cell
  contribution percentages.
- For each variable, compute global average ranks, remove the nuisance design
  by ordinary least squares, and correlate the two residual vectors. The
  nuisance design contains an intercept, chemical reference indicators, and
  neural-date membership fractions (equal weight over a strain's available
  dates). Its rank is 12, leaving 94 residual dimensions.
- Genus adjustment is a sensitivity check that changes the comparison to
  available within-genus variation. Design rank becomes 40, leaving 66 residual
  dimensions. Seventeen strains supply no residual information under this
  model. It is not a correction for every taxonomic or phylogenetic dependency.

The screen covers 162 × 13 = 2,106 compound–cell pairs. Display ranking uses the
minimum absolute rho across six reference/date-adjusted checks, requiring all
six signs to agree: coefficient, unit coefficient, both with genus adjustment,
pre-gate coefficient from the same template, and the separately fitted
unfiltered coefficient. The twelve highest scores are displayed. Agreement
across these six checks is part of selection, not an independent confirmation.
The data used to rank candidates are also the data shown in the figures.

Additional tables check SNR thresholds 0.25 and 0.75, ranks computed within each
reference, associations within each reference, and removal of each reference,
genus, or recording date in turn. Subsets are reranked and the nuisance model
is refitted. Deletion ranges are not confidence intervals or held-out prediction.
No p-values or formal significance claim are made. No new neural templates,
embeddings, or raw-recording processing were run.

## Incompletely reported features

The other 218 features were examined separately using only strains with an
actual original report value. A feature requires at least 40 reported strains
and 25 reference/date residual dimensions to be screened; coverage and QC are
retained in the table. These features are not pooled into the primary PDF.

A focused secondary check uses at least 80 reports, QC RSD <= 0.30, and at least
25 genus-adjusted residual dimensions, then selects the top three using the
same six-check score. Cytosine–ADL (n = 98, rho -0.358) and Guanosine–ADF
(n = 81, rho -0.312) retain their overall negative direction when one reference
is removed. Dopamine–ASJ (n = 83, rho -0.392) is less consistent: removing A306
reduces rho to -0.036. Its A250 subset has only four strains and cannot carry
an interpretation.

Thymidine–ASH is retained as a counterexample: with 104 reported strains its
reference/date-adjusted rho is +0.501, but adding genus reduces it to +0.030.
This is evidence that a prominent pooled pattern may reflect group structure;
it does not identify the cause of that structure. Coverage-specific selection
and missing reports remain limitations for all secondary results.

## Files and reproduction

- `tables/primary_associations.csv`: all primary compound–cell combinations,
  representations, adjustment definitions, and correlations.
- `tables/candidate_ranking.csv`, `selected_candidates.csv`: the exact scoring
  values and displayed selection.
- `tables/candidate_points.csv`: every point used in the twelve relationship
  pages, with strain, reference, genus, original values, and rank residuals.
- `tables/within_reference_associations.csv`, `leave_one_group_out.csv`,
  `within_reference_rank_sensitivity.csv`: subgroup and sensitivity checks.
- `tables/secondary_reported_only_associations.csv`,
  `secondary_context_candidates.csv`, `secondary_context_checks.csv`: secondary
  screening, selection, and subgroup checks with sample counts.
- `tables/candidate_chemical_covariation.csv`: the five other primary chemicals
  most strongly correlated with each displayed compound after reference/date
  adjustment.
- `tables/sample_context.csv`, `neural_date_weights.csv`: analysis alignment
  and fractional recording-date membership.
- `parameters.json`: definitions, design diagnostics, template sign checks,
  and SHA256 hashes of inputs and existing notebooks.
- `verification.json`, `figures/plot_verification.json`: numerical/data-integrity
  checks and plot/PDF checks.

Source chemistry is in `reports/population_first_20260930/tables/`; source
neural coefficients and existing sensitivity fits are in
`reports/exploration_response_profiles_individual_snr_20261002/tables/`.
Source data and notebooks were unchanged, verified by hashes. An independent
read-only review reproduced principal correlations, the 162-feature eligibility
rule, and the displayed selection; PDF checks confirmed page counts, labels,
sample counts, and agreement of residual scatter correlations with the tables.

The small functions in `code/explore_relations.py` and `code/check_secondary.py`
can be called from an existing notebook with the repository's
`.pixi/envs/default/bin/python` environment. Each takes `repo` and `out` paths;
calling it recomputes and overwrites only this exploration's output tables.
`code/plot_relation_candidates.py` reads the saved tables to make the PDF and
preview images. No full notebook execution is needed.

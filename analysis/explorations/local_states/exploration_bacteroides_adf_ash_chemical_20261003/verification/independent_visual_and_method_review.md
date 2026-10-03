# Independent review

The reviewer modified only this `verification/` directory. No new scientific
candidate, model, threshold, cohort exclusion, or sensitivity analysis was added.

`independent_model_check.py` independently rebuilds chemistry from the six original
input files for the full cohort and all 45 held-out folds. It uses NumPy least
squares and SciPy correlation-distance clustering, without importing analysis
functions. Training/test IDs, feature scaling, cluster memberships, family
weights, primary-only candidate selection, all 16 response fits, predictions,
training-mean baselines, scalar/vector errors, aggregate metrics and selection
overlap agree with the saved results. Input, frozen API and protocol hashes match.

`independent_diagnostics_check.py` independently recomputes the saved rank thirds,
all 18 associations (including unnormalized ADF and ASH separately), fixed-axis
deletions, within-label centering, group support, leverage and annotation
composition. It verifies copied model outputs and source hashes.

All three diagnostic PNGs were visually inspected. The held-out plots show
substantial shrinkage and individual error, rather than concealing them. The
rank-third panel shows every strain and overlapping low/middle groups. The
supporting heatmap retains all 29 strains and 13 signed unit coordinates in
chemical-score order, with species, date sets and six flags. Figures use English
labels, readable legends and consistent coordinate scales; no material clipping,
label collision or scientific presentation problem was found.

The scientific interpretation is bounded appropriately after the integrated
report's wording clarifications: positive held-out improvement is conditional
internal evidence, not independent experimental confirmation; rank thirds
describe an already response-selected full-data state; date-set demeaning is a
limited descriptive check; fixed-axis deletion stability does not repeat model
discovery; the 13-coordinate result uses one selected chemical score with 13
separate OLS outputs and no prediction renormalization. The 14-annotation state
does not establish a small causal chemical mechanism or a uniformly continuous
response law.

No unresolved numerical or methodological implementation issue was found.

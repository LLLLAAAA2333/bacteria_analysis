# Bacteroides-local chemical axes

Fixed before local axis calculation, 2026-10-03. Only the chemical branch is covered here. The parent analysis owns neural direction estimation, axis selection and nested training/held-out evaluation; no neural values are read in this branch.

## Data and fixed method

Read only `fresh_chemical_log2.csv`, `fresh_feature_metadata.csv`, `sample_context.csv` from `reports/exploration_chemical_pattern_direct_report_20261003/tables`. Select the 29 strains labeled Bacteroides by exact source genus. Preserve all source species, dates and taxonomy_note; define `taxonomy_flag` only as a nonempty taxonomy_note, without interpreting the note as an exclusion. Source feature names are report-annotation identifiers; family is the provided Mass-column identifier.

For each fit, use ONLY the supplied training strains. Compute each annotation's training mean and sample SD (ddof=1) of log2(c / 1 ng/mL), with no +1 or imputation. Features with SD <=1e-12 log2 units are numerical constants and excluded from standardized clustering, but preserved in the source/scale tables. Every other feature remains; standardization can amplify small raw differences, so raw log2 values, ranges and SDs are exported. Require at least two training strains for a fit.

Compute annotation Pearson correlations across the training strains, average linkage on distance 1-r, one fixed distance cut 0.5. Clusters need at least 3 annotations AND 3 distinct Mass-column families to define an axis; retain every other annotation as ungrouped. No alternative distance, cut, axis type, PCA, hand-picked descriptor, or fallback is searched. Average-linkage cut 0.5 does not imply every within-cluster pair has r>=0.5.

Assign new IDs L01, L02, ... by decreasing member count with the alphabetically smallest member breaking ties. These IDs refer only to the current fit and do not imply correspondence to full-cohort modules or IDs in another training fold. Module weights equal-weight families, then equal-weight annotations within a family. An axis score is the weighted mean of member standardized log2 values. The sign is the arithmetic direction of member values, not flipped to align with another measurement.

Full29 modules are descriptive outputs only. Every training fold must call `fit_axes` again with its training data. Held-out samples must enter only `transform_axes`, which uses that fit's means, scales, members and weights. Returning zero axes is allowed and returns an n-by-0 DataFrame; this branch does not invent a backup predictor.

## API contract

`fit_axes(log_frame, feature_metadata) -> dict`: a pure function, no file reads/writes. log_frame is a finite numeric DataFrame with unique strain index and metabolite columns; metadata is a DataFrame with a unique metabolite index or metabolite column and a nonempty family value for every input feature. Returns means/scales (Series for all input features), retained_features/excluded_features/source_features/train_ids (lists), module_members (insertion-ordered dict by L ID), score_weights (dict of member-indexed Series), train_scores (same strain index, L-ID column order), feature_order, parameters. Fitted objects belong to their own training fold.

`transform_axes(log_frame, fitted) -> DataFrame`: validates finite input and all source features, preserves supplied row order, outputs identical L-ID columns/order to fitted train_scores. Never refits or changes fitted objects. No-module fit transforms to the same n-by-0 schema.

## Saved outputs and plot

Save full29 raw and standardized values, means/scales and raw log2 ranges, context/flags, all module members/family weights, all axes/strain scores, module pair correlations and composition, top-three representatives ranked by member correlation with its own axis, and ungrouped/constant feature tables. Representatives are lookup labels, not functional names. Record source SHA256, code/protocol versions and precise row/column orders.

The single main heatmap uses every local module and all 29 strains. Strain order uses average linkage of all retained standardized chemical features (Euclidean); axis order uses correlation distance between their strain profiles. Neither ordering uses neural values, species or dates. Figure labels and captions are English, and raw-scale data remain available for interpreting standardized colors.

## Verification and limits

Independently recompute scales, scores and module partitions, test transform(train)==train_scores, held-out transformation with fixed training means, feature/row alignment, constant and no-module handling, and input immutability. Do not treat full29 chemistry-only correlation or module definition as external validation. This is a 29-strain within-genus description, subject to species/culture/measurement structure, possibly redundant annotations, panel selection and small sample limitations; it establishes no pathway or causal mechanism. Exact fold-wise model validation belongs to the parent analysis.

Analysis entry writes only to a new output directory and refuses existing results. Use small callable functions from an existing Notebook; do not create or execute a Notebook or install dependencies.

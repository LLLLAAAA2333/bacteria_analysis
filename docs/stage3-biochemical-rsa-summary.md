# Stage 3 Biochemical RSA Summary

Date: 2026-04-02  
Status: method and interpretation summary for ongoing Stage 3 use across datasets

## What Stage 3 Is Trying To Answer

Stage 3 asks a different question from Stage 1 and Stage 2.

- Stage 1 asks whether stimulus identity is reproducibly encoded.
- Stage 2 asks what the pooled neural geometry looks like.
- Stage 3 asks whether that neural geometry is better explained by specific biochemical model spaces.

In practical terms, Stage 3 compares:

- a neural RDM from Stage 2
- one or more biochemical model RDMs built from `matrix.xlsx` plus curated model definitions

The main readout is not "can we classify stimuli?" but:

> which biochemical model geometry is most aligned with the neural geometry?

This is a representational comparison stage. It does not by itself prove mechanism or causality.

## Inputs Used By Stage 3

Stage 3 combines three input families.

### 1. Neural RDMs from Stage 2

These are the pooled neural representational distance matrices for the selected views.

Current default view roles are:

- `response_window`: primary view
- `full_trajectory`: sensitivity view

Stage 3 uses the Stage 2 pooled RDMs as the neural side of the comparison. This means Stage 3 inherits the strengths and weaknesses of the Stage 2 pooled geometry.

### 2. Biochemical matrix and model space definitions

These inputs define the candidate biochemical geometries.

They include:

- the metabolite matrix workbook
- the stimulus-to-sample mapping
- the curated model registry
- the curated model-membership table

Conceptually:

- `global_profile` means "use the whole resolved biochemical profile"
- a curated subset model such as `bile_acid` means "use only a selected feature family"

Each model produces a model RDM in the same stimulus space as the neural RDM.

## How The Metabolism Models Are Built

The metabolism side of Stage 3 is not a single fixed matrix copied directly from `matrix.xlsx`.

Instead, Stage 3 builds each biochemical model in a few explicit steps so the neural-to-metabolite linkage stays stable across datasets.

### 1. Start from the metabolite matrix

The starting table is `matrix.xlsx`.

Its structure is:

- rows: `sample_id`
- columns: metabolite features
- values: metabolite abundance-like measurements

This workbook is treated as the upstream feature source, not as a finished model space by itself.

### 2. Use an explicit stimulus-to-sample mapping

The neural data and the metabolite matrix are joined through a reviewed mapping table.

That mapping contains:

- `stimulus`
- `stim_name`
- `sample_id`

This step matters because Stage 3 does not compare neural stimuli to arbitrary matrix rows. It compares neural stimuli to the exact mapped `sample_id` rows in the metabolite matrix.

So before any model is built, Stage 3 fixes a common stimulus order:

- neural side: stimulus order from the mapping
- model side: matrix rows reindexed into that same stimulus order

### 3. Define which metabolites belong to which model

Model definition is split into two layers.

- `model_registry`: says what each model is
- `model_membership`: says which metabolites belong to that model

This separation is important because it keeps chemistry decisions out of the RSA code.

In practice:

- `global_profile` is the full-profile model and uses all metabolite columns
- a curated subset model such as `bile_acid` uses only the metabolites assigned to that model in `model_membership`

So a metabolism model is not "all metabolites with a label attached." It is:

> a selected subset of matrix columns, applied to the mapped stimulus rows, with a declared feature transformation and distance rule

### 4. Build one feature matrix per model

For each model, Stage 3 builds a stimulus-by-feature matrix.

Its shape is:

`n_stimuli x p_model`

where:

- `n_stimuli` is the number of mapped neural stimuli included in the current run
- `p_model` is the number of retained metabolite features for that model

This is the actual metabolism model object used for RSA.

### 5. Preprocess features according to the model type

Stage 3 does not send raw matrix values directly into the distance calculation.

For `continuous_abundance` models, the current contract is:

1. take the selected metabolite columns
2. apply `log1p`
3. drop zero-variance features within the current stimulus panel
4. z-score each retained feature column

Meaning:

- `log1p` compresses large abundance differences
- zero-variance columns are removed because they do not contribute geometry
- z-scoring keeps one metabolite with a large absolute scale from dominating the whole model

For `binary_presence` models, the contract is:

1. threshold values into `0/1`
2. drop zero-variance columns
3. keep the binary feature matrix

So the metabolism model is really a processed feature space, not just a named metabolite list.

### 6. Compute a model RDM from the feature matrix

Once the feature matrix is built, Stage 3 converts it into a square model RDM.

The distance rule depends on the registry entry:

- `correlation` distance for continuous-abundance models
- `euclidean` distance when explicitly requested
- `jaccard` distance for binary-presence models

This produces a final:

`stimulus x stimulus`

model RDM that can be compared directly with the neural RDM.

### 7. Apply QC before interpretation

The model side is also quality-controlled before interpretation.

Important checks include:

- are all mapped `sample_id` rows present in `matrix.xlsx`
- are all referenced metabolites present in the matrix
- after preprocessing, does the model still retain enough informative features

Primary curated models that retain too few informative features are excluded from primary ranking rather than being treated as fully valid competitors.

This means a metabolism model only enters the main Stage 3 ranking if it is both:

- chemically defined
- numerically usable

## What Different Metabolism Models Mean

Because the model construction is explicit, the meaning of each model is also explicit.

### Global profile model

`global_profile` asks:

> if we use the whole measured metabolite profile, how similar are the stimuli to each other?

This is the broadest biochemical baseline model.

### Curated subset model

A curated subset model such as `bile_acid` asks:

> if we restrict the feature space to one chemically defined family, does that restricted geometry better match the neural geometry?

This is more specific and usually more interpretable, but it also depends more heavily on correct curation and sufficient retained feature count.

### Why this matters for future datasets

This construction makes Stage 3 portable across future data updates.

If later datasets change:

- the set of mapped stimuli
- the exact metabolite columns
- the curated family memberships

the Stage 3 logic stays the same:

1. map neural stimuli to matrix rows
2. build model-specific feature matrices
3. convert them into model RDMs
4. compare those RDMs to the neural geometry

That is the main reason the model space is defined through registry and membership files rather than through hard-coded feature lists inside the analysis logic.

### 3. Optional prototype supplementary inputs

If Stage 3 is run with preprocessing outputs, it can also build prototype-level supplementary analyses.

These prototype supplements do not replace the main pooled RSA result. They answer a narrower question:

> if we aggregate trial responses into per-stimulus prototype vectors, what model geometry do those prototype RDMs resemble?

This supplementary layer is useful for visualization and for checking whether the pooled result hides date-specific structure.

## What A Model RDM Means

A model RDM is a pairwise distance matrix between stimuli in a biochemical feature space.

The exact distances depend on the model definition:

- full-profile models use all resolved metabolites
- curated models use only their assigned metabolite subset

If two stimuli are biochemically similar under a model, their model-RDM distance will be small.

If the neural RDM and a model RDM have similar geometry, Stage 3 will report a higher RSA similarity for that model.

## How RSA Is Computed

Stage 3 compares a neural RDM and a model RDM by aligning their shared upper-triangle entries and computing a rank-based similarity.

The key logic is:

1. take one neural RDM and one model RDM
2. align them on shared stimulus labels
3. flatten the upper triangles
4. compute Spearman similarity between the two distance vectors

Interpretation:

- larger RSA similarity means the two geometries are more aligned
- smaller RSA similarity means the model explains less of the neural geometry

This is a geometry-level comparison. It does not mean the model captures single-neuron responses or trial-level decoding performance.

## What The Main Stage 3 Outputs Mean

The main Stage 3 outputs should be read in layers.

### 1. Ranked primary-model RSA

This is the top-level summary for the formal model comparison.

It answers:

> among the currently eligible primary biochemical models, which one best matches the pooled neural geometry?

This is the most direct Stage 3 answer.

### 2. Neural-versus-top-model RDM figures

These figures are the structural follow-through for the ranked result.

They show:

- neural RDM on the left
- top model RDM on the right

These figures are important because a model can rank first numerically while still matching only part of the neural structure. The paired RDM display lets you inspect where the agreement is broad, local, or only approximate.

### 3. Leave-one-stimulus-out robustness

This is the sensitivity check for the primary view.

It asks:

> does the model preference depend on one or two specific stimuli, or is it relatively stable when each stimulus is removed in turn?

If a model remains directionally similar after removing individual stimuli, that increases confidence that the Stage 3 conclusion is not dominated by a single label.

### 4. Cross-view comparison

This checks whether the Stage 3 story is similar across the chosen neural views.

It asks:

> does the model ranking or neural-model alignment persist when the neural representation is defined using a different time window?

This should be read as sensitivity analysis, not as a second independent experiment.

## What The Prototype Supplement Adds

The prototype supplement is useful, but it should be interpreted as supplementary rather than formal by default.

There are two complementary parts.

### Per-date prototype RSA and paired comparison figures

These outputs ask:

> within each date, if we aggregate repeated trials into stimulus prototypes, which model best matches that date's prototype geometry?

This is the best tool for spotting date-specific shifts in model preference.

The paired per-date comparison figures are especially useful because they show:

- left: per-date neural prototype RDM
- right: that date's top-matching model RDM

This makes it easier to see whether a date-specific model preference reflects a broad geometry match or only a small set of local relationships.

### Pooled prototype RDM figures

These are descriptive pooled prototype displays.

They are useful for visual inspection, but they are not automatically strong evidence of cross-date reproducibility.

If different dates contribute disjoint or weakly overlapping stimulus panels, a pooled prototype figure may still be visually coherent while saying little about true cross-date stability.

So these figures should be used as:

- descriptive geometry displays
- supplements to the formal Stage 3 pooled RSA

not as standalone proof of cross-date replication.

## How To Read Stage 3 Without Overfitting To One Dataset

Because Stage 3 will be rerun on future datasets, the interpretation should focus on structure rather than on one frozen winner.

The most stable reading pattern is:

1. identify the top eligible primary model
2. check whether the runner-up is close or clearly worse
3. inspect the paired RDM figure for the top model
4. check leave-one-stimulus-out robustness
5. check whether the same story roughly persists across neural views
6. use prototype supplementary outputs to see whether the pooled result hides date-level heterogeneity

This lets you interpret Stage 3 as a layered evidence chain rather than as a single ranking table.

## What Stage 3 Can Support

When the signal is clean, Stage 3 can support statements like:

- the pooled neural geometry is more consistent with one biochemical model family than another
- the same model preference is visible across multiple neural views
- the geometry match is stable to leaving out individual stimuli
- prototype-level supplementary analyses are consistent with the pooled summary

These are meaningful representational claims.

## What Stage 3 Cannot Support By Itself

Stage 3 does not by itself support:

- causal claims about which metabolites generate the neural response
- claims that one biochemical family is the only biologically relevant explanation
- claims that a pooled result is stable across individuals or dates without checking overlap and support
- claims that a visually clean paired RDM means the model is mechanistically correct

Stage 3 is best treated as a model-comparison layer built on top of Stage 2 geometry, not as a final biological proof.

## Main Interpretation Risks

### 1. Incomplete model space can make the winner look stronger than it is

If only a small number of primary models are currently populated, the "top model" should be read as:

> top among currently implemented and eligible models

not as:

> globally optimal biochemical explanation

### 2. Pooled geometry can overstate confidence if group overlap is weak

Stage 3 uses pooled neural RDMs by default.

If the pooled Stage 2 geometry is much stronger than the group-level stability evidence, Stage 3 can still produce convincing-looking model matches that should be interpreted cautiously.

### 3. Prototype pooled figures can look more stable than the data really are

If prototype pooled stimuli are effectively date-specific, the pooled prototype displays are still useful visually, but they should remain descriptive unless cross-date support is clearly adequate.

## Aggregation In The Prototype Supplement

Prototype supplementary outputs now support:

- `mean` aggregation
- `median` aggregation

The default remains `mean`.

This choice affects only the prototype supplementary branch, not the main pooled Stage 3 RSA.

How to interpret the two options:

- `mean` is the default descriptive average prototype
- `median` is a robustness-oriented sensitivity check against outlier-sensitive trial aggregation

If the date-level prototype story is similar under both `mean` and `median`, confidence increases that the prototype interpretation is not being driven by a small number of unusual trials.

## Recommended Reporting Pattern

For future datasets, I would report Stage 3 in this order:

1. primary pooled RSA ranking in `response_window`
2. sensitivity check in `full_trajectory`
3. paired neural-versus-top-model RDM figures
4. leave-one-stimulus-out robustness
5. prototype per-date supplementary comparisons
6. pooled prototype figures as descriptive context only

This order keeps the formal pooled comparison separate from the more exploratory prototype layer.

## Main Files To Inspect

Core summary outputs:

- `results/<run>/stage3_rsa/run_summary.json`
- `results/<run>/stage3_rsa/run_summary.md`

Main formal tables:

- `results/<run>/stage3_rsa/tables/rsa_results.parquet`
- `results/<run>/stage3_rsa/tables/rsa_leave_one_stimulus_out.parquet`
- `results/<run>/stage3_rsa/tables/rsa_view_comparison.parquet`

Main formal figures:

- `results/<run>/stage3_rsa/figures/ranked_primary_model_rsa.png`
- `results/<run>/stage3_rsa/figures/neural_vs_top_model_rdm__response_window.png`
- `results/<run>/stage3_rsa/figures/neural_vs_top_model_rdm__full_trajectory.png`
- `results/<run>/stage3_rsa/figures/leave_one_stimulus_out_robustness.png`

Prototype supplementary outputs:

- `results/<run>/stage3_rsa/tables/prototype_rsa_results__per_date.parquet`
- `results/<run>/stage3_rsa/figures/prototype_rsa__per_date__response_window.png`
- `results/<run>/stage3_rsa/figures/prototype_rdm_comparison__per_date__response_window.png`
- `results/<run>/stage3_rsa/figures/prototype_rdm__pooled__response_window.png`

## One-Sentence Summary

Stage 3 tests which biochemical model geometry best aligns with the pooled neural geometry, then uses robustness checks and prototype-level supplementary views to determine whether that model preference looks stable, view-consistent, and interpretable rather than accidental.

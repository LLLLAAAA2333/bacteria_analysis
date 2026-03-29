# Whole Analysis Plan

**Project**: C. elegans chemosensory representation of bacterial metabolites  
**Updated**: 2026-03-27  
**Current Status**: Stage 0 completed; Stage 1 not yet started

## Overview

This document is the high-level scientific roadmap for the project.

The core question is not simply whether the stimuli can be clustered, but whether the worm's chemosensory population activity contains a stable representational geometry for bacterial metabolites, and whether that geometry aligns with meaningful biochemical structure.

The project should be advanced in stages. Each stage must produce a clear scientific outcome before moving to the next one.

## Current State

What is already done:

- Raw data have been standardized into analysis-ready outputs
- Trial-level structure has been preserved
- Baseline centering and mild filtering are implemented
- Real-data preprocessing outputs already exist under `data/processed/`

Current preprocessing outputs:

- `data/processed/clean/neuron_segments_clean.parquet`
- `data/processed/trial_level/trial_metadata.parquet`
- `data/processed/trial_level/trial_wide_baseline_centered.parquet`
- `data/processed/trial_level/trial_tensor_baseline_centered.npz`
- `data/processed/qc/preprocessing_report.json`
- `data/processed/qc/preprocessing_report.md`

Verified current data facts:

- `678` trials
- `22` neurons
- `45` time points
- `130` fully-NaN traces removed
- `1` partially-NaN trace retained

This means the project has completed **Stage 0: preprocessing and data standardization**, not Stage 1.

## Stage 0: Data Standardization

**Goal**: Convert raw recordings into stable trial-level analysis objects.

**Status**: Completed

**What this stage does**:

- preserve the trial structure
- generate a clean long-format table
- generate trial-level wide and tensor representations
- standardize baseline centering
- preserve missing neurons and missing time points as `NaN`

**Output**:

- analysis-ready long table
- trial metadata
- wide table
- tensor object
- QC report

**Exit criterion**:

- preprocessing runs successfully on the real dataset
- output shapes and counts match the dataset contract

## Stage 1: Representation Reliability Validation

**Scientific question**: Is there a reproducible neural representation at all?

**Goal**: Show that the same metabolite evokes more similar neural population responses than different metabolites do.

**Inputs**:

- `trial_metadata.parquet`
- `trial_wide_baseline_centered.parquet`
- `trial_tensor_baseline_centered.npz`

**Main tasks**:

1. Define the representation units to compare:
   - full `neuron x time` trajectories
   - ON window representations (`6..15`)
   - post-stimulus representations (`16..44`)
2. Define one or two baseline distance metrics:
   - correlation distance as the default
   - optional Euclidean distance as sensitivity analysis
3. Compute same-stimulus versus different-stimulus distances
4. Run split-half reliability
5. Run leave-one-worm-out and leave-one-date-out validation
6. Run shuffled-label null controls
7. Compare which time window gives the most reliable structure

**Primary outputs**:

- same-stimulus vs different-stimulus distance distributions
- split-half reliability scores
- leave-one-worm-out reliability scores
- time-window reliability comparison

**Exit criterion**:

- same-stimulus distances are consistently smaller than different-stimulus distances
- the result survives worm/date-level resampling

**Why this stage matters**:

If Stage 1 fails, there is no defensible basis for Stage 2 or Stage 3. In that case the right conclusion is that the current dataset does not yet support representational analysis at the desired level.

## Stage 2: Neural Representational Geometry

**Scientific question**: What is the relative geometry among stimuli in neural population space?

**Goal**: Construct the neural representational distance structure instead of relying on clustering as the main claim.

**Inputs**:

- Stage 1-validated trial representations
- reliable distance metric and time window choice from Stage 1

**Main tasks**:

1. Build neural RDMs (`stimulus x stimulus` representational distance matrices)
2. Build:
   - whole-window RDMs
   - ON-window RDMs
   - post-stimulus RDMs
   - sliding-window time-resolved RDMs
3. Compare RDMs:
   - across worms
   - across dates
   - pooled group level
4. Quantify RDM reliability across resampling schemes
5. Use PCA/MDS/dendrogram only as visualization, not as the primary evidence

**Primary outputs**:

- neural RDM heatmaps
- time-resolved RDM series
- RDM reliability curves
- visualization panels for publication

**Exit criterion**:

- the neural geometry is stable enough to interpret
- at least one time window shows reproducible representational structure

**Why this stage matters**:

Stage 2 answers what the neural space looks like before trying to explain what it means.

## Stage 3: Biochemical Meaning / RSA

**Scientific question**: What biochemical or metabolic properties are encoded by the neural geometry?

**Goal**: Link the neural RDM to external model RDMs derived from metabolite properties.

**Inputs**:

- neural RDMs from Stage 2
- metabolite descriptors derived from `data/matrix.xlsx` or a curated chemical-property table

**Main tasks**:

1. Define the model space:
   - metabolic composition distance
   - component overlap or abundance distance
   - manually curated categories if needed
   - bacterial origin or other biologically meaningful grouping
2. Build one or more model RDMs
3. Run RSA between neural RDMs and model RDMs
4. Compare models across:
   - time windows
   - worms
   - resampling folds
5. Evaluate significance with permutation or resampling

**Primary outputs**:

- neural-model RDM correlation table
- time-resolved model-match curves
- ranking of biochemical models by explanatory power

**Exit criterion**:

- at least one model explains the neural geometry better than null/shuffled controls
- the neural-model match is stable across resampling

**Why this stage matters**:

This is the step that turns a mathematical geometry into a biological result.

## Stage 4: Network Dissection and Figure-Ready Visualization

**Scientific question**: Which neurons drive the representational geometry?

**Goal**: Localize the contribution of individual neurons or neuron groups to the neural code.

**Inputs**:

- reliable neural geometry from Stage 2
- validated neural-model relationship from Stage 3

**Main tasks**:

1. Perform leave-one-neuron-out analysis
2. Quantify how much each neuron changes:
   - Stage 1 reliability
   - Stage 2 RDM stability
   - Stage 3 neural-model match
3. Compare left/right pairs where biologically relevant
4. Build publication-quality visualizations:
   - contribution ranking plots
   - dendrograms
   - MDS/PCA visualizations
   - heatmaps aligned to metadata

**Primary outputs**:

- neuron contribution ranking
- leave-one-neuron-out effect sizes
- polished geometry and contribution figures

**Exit criterion**:

- a small subset of neurons or neuron groups can be identified as major contributors

## Stage 5: Behavioral / Ecological Link

**Scientific question**: Does the neural representational space matter for behavior?

**Goal**: Link neural geometry to attraction/avoidance or other behavioral significance.

**Inputs**:

- outputs from Stages 2 and 3
- behavior labels, chemotaxis measurements, or other external behavioral data

**Main tasks**:

1. Attach behavioral labels to stimuli if available
2. Test whether neural distances predict behavioral similarity
3. If possible, predict response to held-out metabolites
4. Evaluate whether the neural manifold supports generalization

**Primary outputs**:

- neural-behavior relationship analysis
- predictive generalization results

**Exit criterion**:

- behavior can be explained or predicted from the neural representational structure

**Note**:

This stage is highly valuable but is not required before Stage 1 to Stage 4 are complete. It depends on extra data and should be treated as a later milestone rather than an immediate requirement.

## Recommended Execution Order

The next steps should be:

1. Finish Stage 1 reliability validation
2. Only if Stage 1 succeeds, move to Stage 2 neural geometry
3. Only if Stage 2 is stable, move to Stage 3 RSA/model matching
4. Then perform Stage 4 neuron contribution analysis
5. Treat Stage 5 as optional or later, depending on behavior data availability

## What Counts as Success

A strong near-term project success is:

- Stage 1 passes
- Stage 2 reveals a stable neural geometry
- Stage 3 links that geometry to interpretable metabolic structure

That alone already supports a serious scientific story.

Stage 4 strengthens mechanism.
Stage 5 strengthens biological impact.

## Immediate Next Deliverable

The immediate next deliverable should be a **Stage 1 plan**, not a Stage 2 implementation.

That plan should define:

- which representations will be compared
- which distance metrics will be used
- which resampling schemes will be used
- what statistical outputs count as evidence of reliability
- which figures and tables Stage 1 must produce

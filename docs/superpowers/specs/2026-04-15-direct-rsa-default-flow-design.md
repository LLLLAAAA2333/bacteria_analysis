# Direct RSA Default Flow Design

Date: 2026-04-15
Status: Draft for user review
Topic: Make RSA run directly from preprocess outputs and `matrix.xlsx` by default while keeping curated model-space inputs as an advanced optional path

## Goal

Refactor the Stage 3 RSA entry path so the default user workflow becomes:

1. `preprocess`
2. `reliability`
3. `rsa`

The default RSA path should no longer require a pre-generated `model_space` directory when the user only wants the `global_profile` model. Instead, RSA should be able to consume:

- preprocess outputs
- `data/matrix.xlsx`

and construct the minimum required model-side objects in memory at runtime.

## Motivation

The current Stage 3 structure exposes `model_space` as an explicit user-facing prerequisite even when the user only wants a single full-matrix baseline model. That is heavier than necessary.

For the current user goal:

- the main scientific question is the association between neural geometry and the full metabolite profile
- curated subset models are not the default focus
- `matrix.xlsx` is already the user-provided metabolite data source
- forcing `rsa` to depend on pre-generated intermediate CSVs makes the workflow feel indirect and harder to reason about

This refactor should make the common path direct while preserving room for later curated biochemical model work.

## Non-Goals

- Do not remove support for curated `model_space` inputs.
- Do not redesign the Stage 3 RSA statistics, permutation logic, or figure logic.
- Do not remove `build_model_space.py`.
- Do not implement shared metabolite cache generation in this slice.
- Do not redesign subset-model curation, annotation, or membership review workflows.
- Do not change the scientific interpretation boundary between Stage 2 geometry validation and Stage 3 RSA.

## Current Problem

Today `scripts/run_rsa.py` resolves Stage 3 model inputs from a directory that contains:

- `stimulus_sample_map.csv`
- `metabolite_annotation.csv`
- `model_registry.csv`
- `model_membership.csv`

This is appropriate when running multiple curated or exploratory biochemical models, but it is unnecessarily heavy for the common baseline case where the user only wants `global_profile`.

In practice, the default path currently exposes an implementation detail:

- the user has the authoritative metabolite matrix in `matrix.xlsx`
- the preprocess outputs already contain the authoritative neural stimulus metadata
- but RSA still requires an intermediate model-side directory to be created first

That extra step makes the pipeline harder to use and obscures the fact that `global_profile` does not need curated subset membership.

## Design Summary

Keep two RSA input modes, but make only one of them the default:

### Default mode: direct RSA

Inputs:

- `--matrix`
- `--preprocess-root`

Behavior:

- derive `stimulus x stim_name x sample_id` from preprocess metadata
- read `matrix.xlsx`
- construct the `global_profile` model in memory
- build the model RDM directly from the mapped matrix rows
- run RSA without requiring pre-generated `model_space` files

### Advanced mode: curated model-space RSA

Input:

- `--model-input-root`

Behavior:

- load the existing curated Stage 3 model input CSV contract
- preserve support for curated subset models and future custom model families

The direct mode becomes the main user-facing path. The curated model-space path remains available for advanced usage and future expansion.

## User-Facing Workflow

The intended default workflow after this refactor is:

1. run preprocessing
2. run reliability
3. run RSA directly from preprocess outputs and `matrix.xlsx`

The user should not need to generate a `model_space` directory just to ask whether neural geometry aligns with the full metabolite profile.

## CLI Contract

### Argument defaults

- `--matrix`
  - default: `data/matrix.xlsx`
  - always optional if the default path exists
- `--preprocess-root`
  - default: `None`
  - required for the recommended direct path
- `--model-input-root`
  - default: `None`
  - optional advanced override
- `--geometry-root`
  - default: existing geometry-root resolution behavior
  - compatibility-only fallback input

Direct mode is selected by omitting `--model-input-root` entirely. The implementation does not need to support an explicit empty string sentinel for direct mode selection.

### Recommended default usage

```text
pixi run rsa --preprocess-root <preprocess_root> --matrix data/matrix.xlsx --output-root <results_root>
```

### Advanced usage

```text
pixi run rsa --preprocess-root <preprocess_root> --matrix data/matrix.xlsx --model-input-root <model_space_root> --output-root <results_root>
```

### Legacy fallback usage

```text
pixi run rsa --geometry-root <geometry_root> --matrix data/matrix.xlsx --model-input-root <model_space_root> --output-root <results_root>
```

## CLI Rules

- `--preprocess-root` is the primary default input root for Stage 3 RSA.
- `--matrix` remains the metabolite-side input.
- `--model-input-root` must change from a default path-based argument to a `None`-default optional argument.
- `--model-input-root` remains supported, but is no longer the primary default workflow.
- If `--model-input-root` is omitted, RSA must enter direct mode automatically.
- If `--model-input-root` is provided, RSA must load curated model inputs from that directory.
- If both `--preprocess-root` and `--model-input-root` are provided:
  - neural-side construction still comes from `--preprocess-root`
  - model-side inputs come from `--model-input-root`

## CLI Precedence Table

The implementation should use these exact mode rules:

| Inputs | Mode | Neural Side | Model Side | Status |
|--------|------|-------------|------------|--------|
| `--preprocess-root` and no `--model-input-root` | direct mode | build neural RDMs from preprocess outputs | synthesize `global_profile` in memory | recommended default |
| `--preprocess-root` and `--model-input-root` | curated mode | build neural RDMs from preprocess outputs | load curated model inputs | supported advanced path |
| no `--preprocess-root`, but `--model-input-root` | legacy curated fallback | load pooled neural RDMs from `--geometry-root` or its legacy default | load curated model inputs | temporary compatibility path only |
| no `--preprocess-root` and no `--model-input-root` | invalid | none | none | fail fast |

Additional precedence rules:

- If `--preprocess-root` is provided, RSA must not require `--geometry-root` to run.
- If `--preprocess-root` is provided, any geometry-root path is ignored for core RSA execution.
- Geometry-root based pooled neural inputs remain available only to preserve older curated workflows while the new direct path is adopted.
- Help text and docs should no longer position geometry-root fallback as the recommended path.

## Neural Input Contract

Direct mode must define its neural-side input contract explicitly.

### Required direct-mode neural inputs

`--preprocess-root` must contain:

- `trial_level/trial_metadata.parquet`
- `trial_level/trial_tensor_baseline_centered.npz`

Optional:

- `trial_level/trial_wide_baseline_centered.parquet`

These are the same preprocess artifacts already used by the aggregated-response Stage 3 path.

### Direct-mode neural behavior

When `--preprocess-root` is provided, RSA must:

- load aggregated-response context inputs from preprocess outputs
- build pooled neural RDMs at runtime for:
  - `response_window`
  - `full_trajectory`
- use those pooled neural RDMs as the neural-side Stage 3 inputs

Direct mode must not depend on Stage 2 geometry parquet outputs.

### Legacy fallback neural behavior

Only when `--preprocess-root` is absent and `--model-input-root` is present may RSA fall back to pooled neural RDMs loaded from:

- `--geometry-root`
- or the existing legacy default geometry root

This fallback exists for temporary compatibility and should not define the new default workflow.

## Runtime Data Flow

### Direct mode

1. read preprocess outputs from `--preprocess-root`
2. load aggregated-response context inputs from:
   - `trial_level/trial_metadata.parquet`
   - `trial_level/trial_tensor_baseline_centered.npz`
   - optional `trial_level/trial_wide_baseline_centered.parquet`
3. derive a unique `stimulus x stim_name x sample_id` mapping
4. read `matrix.xlsx`
5. validate that all mapped `sample_id` values exist in the matrix
6. synthesize an in-memory Stage 3 model input bundle containing:
   - stimulus-to-sample mapping
   - minimal metabolite annotation covering all matrix columns
   - minimal model registry containing `global_profile`
   - minimal model membership containing all matrix metabolites for `global_profile`
7. resolve model inputs
8. construct the `global_profile` model RDM
9. build pooled neural RDMs at runtime from preprocess-derived aggregated responses
10. run RSA and write the usual outputs

### Curated mode

If `--preprocess-root` is provided:

1. read preprocess outputs from `--preprocess-root` for the neural side
2. read `matrix.xlsx`
3. load `--model-input-root`
4. resolve curated model inputs as today
5. run RSA using preprocess-derived pooled neural RDMs

If `--preprocess-root` is absent:

1. read pooled neural RDMs from `--geometry-root` or the legacy default geometry root
2. read `matrix.xlsx`
3. load `--model-input-root`
4. resolve curated model inputs as today
5. run RSA unchanged

## Direct-Mode Input Synthesis

The direct path should synthesize only the minimum objects required by the existing Stage 3 machinery.

### Stimulus mapping

Reuse the same deterministic mapping rule already established by the current model-space builder:

- read `stimulus` and `stim_name` from `trial_metadata.parquet`
- require a one-to-one `stimulus -> stim_name` mapping
- derive `sample_id` from the first token of `stim_name`
- require final `sample_id` values to be unique
- require all derived `sample_id` values to exist in `matrix.xlsx`

This keeps the mapping rule explicit and consistent with the existing Stage 3 assumptions.

### Minimal metabolite annotation

Direct mode should synthesize a full-coverage annotation table in memory with one row per matrix metabolite and neutral default fields.

Purpose:

- satisfy the existing Stage 3 input contract
- avoid requiring a user-facing annotation CSV when only `global_profile` is being run

This synthesized annotation is a runtime compatibility object, not a reviewed biochemical truth table.

### Minimal model registry

Direct mode should synthesize one registry row:

- `model_id = global_profile`
- `model_tier = primary`
- `model_status = primary`
- `feature_kind = continuous_abundance`
- `distance_kind = correlation`

This matches the current Stage 3 baseline model behavior.

### Minimal model membership

Direct mode should synthesize `global_profile x metabolite_name` membership for every metabolite column in `matrix.xlsx`.

No subset models are created in direct mode unless the user explicitly provides a curated `--model-input-root`.

## Error Handling

Direct mode must fail clearly when:

- `--preprocess-root` is missing or malformed
- `trial_metadata.parquet` does not contain enough information to derive the mapping
- one `stimulus` maps to multiple `stim_name` values
- derived `sample_id` values are duplicated
- derived `sample_id` values are absent from `matrix.xlsx`
- `matrix.xlsx` is structurally invalid
- `global_profile` does not retain enough valid features to build a meaningful model RDM

Curated mode should retain the existing validation rules for:

- model registry
- membership
- annotation
- matrix alignment

## Compatibility Strategy

This refactor should preserve backward compatibility at the code boundary even while changing the recommended workflow.

### Preserve

- `src/bacteria_analysis/model_space.py`
- `resolve_model_inputs(...)`
- `build_model_space.py`
- the existing curated Stage 3 model input CSV contract
- support for future curated subset models

### Change

- `scripts/run_rsa.py` default behavior
- the default user-facing CLI story
- internal runtime assembly of Stage 3 inputs for the `global_profile` case

### De-emphasize

- the requirement to pre-generate `model_space` for baseline RSA
- Stage 2 geometry outputs as the default Stage 3 runtime dependency

## Architectural Recommendation

Add a small direct-input assembly layer rather than overloading existing builder modules.

Recommended additions:

- a helper in `src/bacteria_analysis/model_space.py` or a small adjacent module to build direct-mode Stage 3 inputs from:
  - preprocess metadata
  - matrix path
- minimal CLI branching in `scripts/run_rsa.py`

Recommended principle:

- reuse existing validation and RDM-building code
- only replace the source of model inputs, not the RSA core

## Testing Strategy

Add deterministic coverage for:

- direct-mode RSA with `--preprocess-root + --matrix` and no `--model-input-root`
- correct synthesis of the mapping from preprocess metadata
- failure on ambiguous or invalid derived `sample_id`
- failure when mapped `sample_id` rows are missing from `matrix.xlsx`
- synthesis of a minimal `global_profile` registry and membership
- parity that direct mode still produces valid RSA outputs
- continued success of curated mode when `--model-input-root` is provided

The first slice should focus on behavior and compatibility, not on broader model-space redesign.

## Recommended First Implementation Slice

The first refactor slice should:

- make direct mode the default RSA path
- require `--preprocess-root` for that path
- synthesize only `global_profile` model inputs in memory
- keep curated `--model-input-root` support intact
- keep current RSA scoring and output writing logic intact
- update CLI help text and tests to reflect the new default workflow

This is enough to align the software with the intended user workflow:

1. `preprocess`
2. `reliability`
3. `rsa`

without forcing users through a `model_space` generation step for the baseline full-profile analysis.

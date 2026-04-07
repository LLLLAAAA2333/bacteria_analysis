# Neutral Analysis Contract Design

Date: 2026-04-04
Status: Approved in chat before implementation
Topic: Remove stage-prefixed outward naming and remove main-versus-supplementary figure hierarchy from default outputs

## Goal

The analysis pipeline is no longer being treated as a paper-ready staged program with a stable main-figure versus supplementary-figure contract.

The goal of this design is to make the executable surface more portable by:

- replacing stage-prefixed public naming with responsibility-based naming
- moving output directories to neutral names
- removing `primary` and `supplementary` wording from default RSA figure and summary contracts
- keeping the internal model registry tier fields intact for compatibility in this change

## Current Context

The repository currently mixes two different concepts:

- milestone language used to describe the historical order of analysis work
- executable contracts used by code, CLI entry points, output paths, tests, and run summaries

That mixing shows up in:

- exported helpers such as `parse_stage2_views()` and `write_stage3_outputs()`
- default result directories such as `stage1_reliability`, `stage2_geometry`, and `stage3_rsa`
- generated markdown summaries titled `Stage 1`, `Stage 2`, and `Stage 3`
- RSA figure names and summary fields such as `ranked_primary_model_rsa`, `top_primary_models_by_view`, and `supplementary_models`

This makes the codebase harder to reuse outside the original staged narrative.

## User-Approved Scope

The approved scope for this implementation is:

1. allow old result directories to migrate to neutral names
2. remove stage language from outward-facing code paths where practical
3. remove `primary` and `supplementary` wording from default figure and summary contracts
4. keep the underlying registry taxonomy and ranking logic for now

The design explicitly does not rename every internal data column in this pass. The `model_tier` and related registry fields remain valid internal compatibility fields until there is a separate decision to redesign the model taxonomy itself.

## Options Considered

### Option 1: Surface Neutralization Only

Rename public helpers, CLI defaults, output directories, summary titles, figure names, and summary keys, while preserving internal tier fields and selection logic.

Pros:

- directly addresses the portability complaint
- lower behavioral risk
- avoids rewriting validated model-ranking logic
- keeps migration focused on executable contracts

Cons:

- some internal naming debt remains
- compatibility aliases are needed during transition

### Option 2: Full Taxonomy Rewrite

Rename both the outward contract and the underlying model registry taxonomy in one pass.

Pros:

- produces a fully neutral naming model

Cons:

- much wider blast radius
- higher risk of breaking ranking and filtering logic
- mixes two separate decisions into one change

## Chosen Approach

Use Option 1.

This implementation should neutralize the outward contract while preserving validated internal behavior. Stage language may remain in historical documents, but executable naming should become responsibility-based or analysis-based.

For RSA outputs, the default contract should stop promoting a main-versus-supplementary hierarchy. Default artifacts should describe ranked models, model comparisons, and available model groups without implying paper figure priority.

## Design Details

### Public Naming

Introduce responsibility-based names for exported helpers and writers. Keep compatibility aliases where needed so existing imports continue to work during the transition.

Examples of the target direction:

- `parse_stage2_views` -> `parse_geometry_views`
- `ensure_stage2_output_dirs` -> `ensure_geometry_output_dirs`
- `write_stage2_outputs` -> `write_geometry_outputs`
- `load_stage2_pooled_neural_rdms` -> `load_geometry_pooled_neural_rdms`
- `run_stage3_rsa` -> `run_biochemical_rsa`
- `ensure_stage3_output_dirs` -> `ensure_rsa_output_dirs`
- `write_stage3_outputs` -> `write_rsa_outputs`
- `ensure_stage1_output_dirs` -> `ensure_reliability_output_dirs`
- `write_stage1_outputs` -> `write_reliability_outputs`

Compatibility aliases should remain exported in this change so current callers do not break.

### Output Directories

Replace staged default directories with neutral names:

- `stage1_reliability` -> `reliability`
- `stage2_geometry` -> `geometry`
- `stage3_rsa` -> `rsa`

CLI defaults, worktree-aware path resolution, smoke tests, and run-summary paths should move together.

### Summary And Figure Contract

Run summaries should stop using stage titles and stop labeling default results as `primary` versus `supplementary`.

Representative changes:

- `# Stage 1 Reliability Run Summary` -> `# Reliability Analysis Run Summary`
- `# Stage 2 Geometry Run Summary` -> `# Geometry Analysis Run Summary`
- `# Stage 3 Biochemical RSA Run Summary` -> `# Biochemical RSA Run Summary`
- `ranked_primary_model_rsa` -> `ranked_model_rsa`
- `top_primary_models_by_view` -> `top_models_by_view`
- `primary_models` -> `ranked_models`
- `supplementary_models` -> `additional_models`
- `prototype_supplement_enabled` -> `prototype_context_enabled`

The internal registry may still use `model_tier=primary/supplementary`; the writer should translate that into neutral outward metadata.

## Testing

Update unit and CLI smoke tests to validate:

- neutral directory names
- neutral exported helper names still work
- compatibility aliases still work
- neutral summary titles and keys are written
- RSA figure names no longer imply main-versus-supplementary hierarchy

## Success Criteria

The change is complete when:

- default CLI outputs land under neutral directory names
- public helpers and writers expose neutral names
- run summaries no longer present Stage 1/2/3 titles
- RSA default figure and summary contracts no longer use `primary` and `supplementary` wording
- targeted tests pass without changing validated ranking behavior

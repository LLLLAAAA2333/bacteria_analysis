# Legacy Analysis Boundary

Date: 2026-04-30

The old Stage 1/2/3 scripts and the later review scripts remain useful audit
and reference material, but they are no longer the preferred source of truth for
new analysis runs.

## Current Boundary

- Stage 1/2/3 outputs are legacy intermediate products.
- Review-output directories under `results/` are legacy result snapshots.
- New core analyses should not read generated review-output directories as
inputs.
- New core analyses should start from raw neural parquet, matrix data, and
chemical metadata through the function-first APIs.
- `docs/ChatGPT-Guidance.md` is background reference only, not current project
policy.

## Preferred Entry Points

Use these Python functions for new work:

- `build_analysis_dataset`
- `build_anchor_dataset`
- `run_anchor_batch_effect`
- `run_rdm_alignment`
- `run_chemical_class_rsa`
- `save_analysis_result`

The notebooks in `notebook/` are thin wrappers around these functions. They are
for inspection and presentation, not for core scientific logic.

## Output Policy

Analysis functions return in-memory `AnalysisResult` objects by default. They do
not write large intermediate directories unless `save_analysis_result(...)` is
called explicitly.

When saved, the default output is limited to final summaries, compact tables,
figures, reported RDMs, and audit metadata. Debug tables are written only when
`include_debug=True` is passed to `save_analysis_result`.

## Frozen Legacy Workflow

After the notebook workflow is validated, `scripts/run_rsa.py` should be treated
as frozen legacy infrastructure. Keep it available for auditability, but do not
extend it for new scientific questions unless a future decision explicitly
reopens the legacy workflow.

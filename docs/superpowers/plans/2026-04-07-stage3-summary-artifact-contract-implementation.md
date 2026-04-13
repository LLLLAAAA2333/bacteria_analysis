# Stage 3 Summary Artifact Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Stage 3 stimulus-sensitivity figure artifact, remove the redundant ranked-model and view-comparison histogram figures, and move their full detail payloads into `run_summary.json` and `run_summary.md`.

**Architecture:** Keep the change local to the Stage 3 output writer in `src/bacteria_analysis/rsa_outputs.py`. Replace the two removed figure outputs with summary-builder helpers that extract stable, full-detail arrays from existing Stage 3 tables, rename the remaining sensitivity artifact in both write paths and cleanup paths, and update summary rendering plus tests to match the new output contract.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, pytest, standard library (`json`, `pathlib`)

---

## Overview

This plan implements the approved spec:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-04-07-stage3-summary-artifact-contract-design.md`

The implementation intentionally does not change:

- Stage 3 RSA calculations in `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
- Stage 3 parquet table schemas
- leave-one-stimulus sensitivity computation
- paired-RDM figure rendering
- aggregated-response generation

This is an output-contract refactor only.

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: rename the public stimulus-sensitivity artifact, stop writing the two removed histogram figures, add summary-detail extraction helpers, update figure cleanup, and extend markdown summary rendering.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
  Responsibility: lock in the renamed figure artifact, the removed figure outputs, the new `run_summary.json` arrays, the markdown summary sections, and stale-figure cleanup.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
  Responsibility: update CLI artifact expectations and summary assertions so the end-to-end Stage 3 contract matches the new writer behavior.

No new production modules are required. Keep the implementation in the existing writer unless a helper extraction is clearly needed to keep `rsa_outputs.py` readable.

## Contract To Implement

After implementation, Stage 3 should behave as follows:

- write `single_stimulus_sensitivity.png`
- stop writing `ranked_model_rsa.png`
- stop writing `view_comparison_summary.png`
- remove the old names from `REQUIRED_FIGURES` and `figure_names`
- expose full ranked-model detail rows in `run_summary.json["ranked_model_rsa_details"]`
- expose full cross-view detail rows in `run_summary.json["view_comparison_details"]`
- mirror both collections in `run_summary.md`
- delete stale legacy figures on successful Stage 3 rewrites:
  - `ranked_model_rsa.png`
  - `view_comparison_summary.png`
  - `leave_one_stimulus_out_robustness.png`

## Task 1: Lock The New Output Contract In Tests

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`

- [ ] **Step 1: Write a failing unit test for the renamed sensitivity figure artifact**

```python
def test_write_rsa_outputs_writes_single_stimulus_sensitivity_figure(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")

    assert (written["figures_dir"] / "single_stimulus_sensitivity.png").exists()
    assert "single_stimulus_sensitivity" in written
    assert not (written["figures_dir"] / "leave_one_stimulus_out_robustness.png").exists()
```

- [ ] **Step 2: Write failing tests that the removed histogram figures are no longer written**

```python
def test_write_rsa_outputs_does_not_write_removed_summary_figures(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")

    assert not (written["figures_dir"] / "ranked_model_rsa.png").exists()
    assert not (written["figures_dir"] / "view_comparison_summary.png").exists()
```

- [ ] **Step 3: Write failing summary tests for the new JSON detail arrays**

```python
def test_write_rsa_outputs_reports_full_summary_detail_arrays(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["ranked_model_rsa_details"][0]["model_id"] == "global_profile"
    assert "rsa_similarity" in summary["ranked_model_rsa_details"][0]
    assert summary["view_comparison_details"][0]["view_name"] == "response_window"
    assert "reference_view_name" in summary["view_comparison_details"][0]
```

- [ ] **Step 4: Write a failing markdown-summary test for the new detail sections**

```python
def test_write_rsa_outputs_markdown_summary_includes_detail_sections(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    markdown = (written["output_root"] / "run_summary.md").read_text(encoding="utf-8")

    assert "## Ranked Model RSA Details" in markdown
    assert "## View Comparison Details" in markdown
    assert "global_profile | view=response_window" in markdown
```

- [ ] **Step 5: Write failing cleanup and CLI smoke tests for the removed legacy figure files**

```python
def test_write_rsa_outputs_removes_stale_summary_figures(tmp_path, synthetic_stage3_outputs):
    output_root = tmp_path / "rsa"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "ranked_model_rsa.png",
        "view_comparison_summary.png",
        "leave_one_stimulus_out_robustness.png",
    ):
        (figures_dir / stale_name).write_bytes(b"stale")

    written = write_rsa_outputs(synthetic_stage3_outputs, output_root)

    assert not (written["figures_dir"] / "ranked_model_rsa.png").exists()
    assert not (written["figures_dir"] / "view_comparison_summary.png").exists()
    assert not (written["figures_dir"] / "leave_one_stimulus_out_robustness.png").exists()
```

CLI smoke assertion update target:

```python
assert summary["figure_names"] == [
    "single_stimulus_sensitivity",
    "neural_vs_top_model_rdm__response_window",
    "neural_vs_top_model_rdm__full_trajectory",
]
assert stage3_root.joinpath("figures", "single_stimulus_sensitivity.png").exists()
assert not stage3_root.joinpath("figures", "ranked_model_rsa.png").exists()
```

- [ ] **Step 6: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "single_stimulus_sensitivity or removed_summary_figures or summary_detail_arrays or markdown_summary_includes_detail_sections or removes_stale_summary_figures" -v`

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -k "writes_rsa_outputs or writes_rsa_prototype_context_outputs" -v`

Expected: FAIL because the current writer still emits the old figure names, still writes the two histogram figures, and does not add the new summary-detail sections.

- [ ] **Step 7: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "test: lock stage3 summary artifact contract"
```

## Task 2: Rename The Sensitivity Artifact And Remove The Redundant Figures

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Rename the public sensitivity artifact throughout the writer**

Implementation targets:

- `REQUIRED_FIGURES`
- `written[...]` assignment in `_write_rsa_artifacts`
- output filename in `figures/`
- empty-figure titles/messages where needed

Minimal code target:

```python
written["single_stimulus_sensitivity"] = _plot_single_stimulus_sensitivity(
    leave_one_out,
    group_summary["ranked_models"],
    dirs["figures_dir"] / "single_stimulus_sensitivity.png",
)
```

- [ ] **Step 2: Replace the old helper name with a stable new helper name**

```python
def _plot_single_stimulus_sensitivity(
    leave_one_out: pd.DataFrame,
    ranked_models: list[str],
    path: Path,
) -> Path:
    ...
```

Implementation requirements:

- keep the current plotting behavior in this round
- only change the public artifact name and title text to `Single-Stimulus Sensitivity`
- do not redesign the plot encoding here

- [ ] **Step 3: Stop writing the two removed summary figures**

Implementation targets:

- delete the `_plot_ranked_model_rsa(...)` call site
- delete the `_plot_view_comparison_summary(...)` call site
- remove both names from `REQUIRED_FIGURES`

Minimal code target:

```python
figure_names = [
    *REQUIRED_FIGURES,
    *_build_neural_vs_model_figure_names(figure_view_names),
    *aggregated_response_summary["aggregated_response_figure_names"],
]
```

Expected `REQUIRED_FIGURES` after this task:

```python
REQUIRED_FIGURES: tuple[str, ...] = ("single_stimulus_sensitivity",)
```

- [ ] **Step 4: Extend stale-figure cleanup to remove the old file names**

Minimal code target:

```python
def _remove_legacy_rsa_figures(figures_dir: Path) -> None:
    for legacy_name in (
        "neural_vs_top_model_rdm_panel.png",
        "ranked_primary_model_rsa.png",
        "ranked_model_rsa.png",
        "view_comparison_summary.png",
        "leave_one_stimulus_out_robustness.png",
    ):
        legacy_figure = figures_dir / legacy_name
        if legacy_figure.exists():
            legacy_figure.unlink()
```

Implementation requirement:

- stale cleanup should run during normal successful Stage 3 rewrites, matching current `write_rsa_outputs` refresh behavior

- [ ] **Step 5: Run the targeted tests to verify the artifact contract now matches**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "single_stimulus_sensitivity or removed_summary_figures or figure_names or removes_stale_summary_figures" -v`

Expected: the renamed figure and removed-file assertions PASS, while the new summary-detail assertions still fail until Task 3 is complete.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "refactor: simplify stage3 summary artifacts"
```

## Task 3: Move Ranked-Model And Cross-View Details Into Run Summary

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Add failing helper-level tests for stable summary detail extraction**

```python
def test_build_ranked_model_rsa_details_filters_to_focus_view_and_ranked_models():
    details = rsa_outputs._build_ranked_model_rsa_details(
        synthetic_stage3_outputs["rsa_results"],
        ranked_models=["global_profile", "bile_acid"],
        focus_view="response_window",
    )
    assert [row["model_id"] for row in details] == ["global_profile", "bile_acid"]
    assert all(row["view_name"] == "response_window" for row in details)


def test_build_view_comparison_details_preserves_input_row_order():
    details = rsa_outputs._build_view_comparison_details(
        synthetic_stage3_outputs["cross_view_comparison"]
    )
    assert [row["view_name"] for row in details] == ["response_window", "full_trajectory"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "build_ranked_model_rsa_details or build_view_comparison_details" -v`

Expected: FAIL because the summary-detail helpers do not exist yet.

- [ ] **Step 3: Implement `_build_ranked_model_rsa_details`**

```python
def _build_ranked_model_rsa_details(
    rsa_results: pd.DataFrame,
    *,
    ranked_models: list[str],
    focus_view: str | None,
) -> list[dict[str, Any]]:
    if rsa_results.empty or not ranked_models:
        return []

    ranked = rsa_results.copy()
    ranked["model_id"] = ranked["model_id"].astype(str)
    ranked["view_name"] = ranked["view_name"].astype(str)
    ranked["rsa_similarity"] = pd.to_numeric(ranked["rsa_similarity"], errors="coerce")
    ranked = ranked.loc[ranked["model_id"].isin(ranked_models)]
    if focus_view is not None:
        ranked = ranked.loc[ranked["view_name"] == focus_view]
    ranked = ranked.loc[np.isfinite(ranked["rsa_similarity"])]
    if ranked.empty:
        return []

    ranked = ranked.sort_values(["rsa_similarity", "model_id"], ascending=[False, True])
    detail_columns = [
        "view_name",
        "model_id",
        "rsa_similarity",
        "p_value_raw",
        "p_value_fdr",
        "n_shared_entries",
        "score_status",
        "is_top_model",
    ]
    available_columns = [column for column in detail_columns if column in ranked.columns]
    return ranked.loc[:, available_columns].to_dict(orient="records")
```

Implementation requirements:

- use `rsa_view_comparison` / `cross_view_comparison` alias resolution already present in the writer
- keep empty-array behavior stable
- never include non-finite `rsa_similarity` rows

- [ ] **Step 4: Implement `_build_view_comparison_details` and wire both arrays into `_build_run_summary`**

```python
def _build_view_comparison_details(view_comparison: pd.DataFrame) -> list[dict[str, Any]]:
    if view_comparison.empty:
        return []

    summary = view_comparison.copy()
    summary["rsa_similarity"] = pd.to_numeric(summary["rsa_similarity"], errors="coerce")
    summary = summary.loc[np.isfinite(summary["rsa_similarity"])]
    if summary.empty:
        return []

    detail_columns = [
        "view_name",
        "reference_view_name",
        "comparison_scope",
        "rsa_similarity",
        "p_value_raw",
        "p_value_fdr",
        "n_shared_entries",
        "score_status",
    ]
    available_columns = [column for column in detail_columns if column in summary.columns]
    return summary.loc[:, available_columns].to_dict(orient="records")
```

Implementation requirements:

- preserve `rsa_view_comparison` row order
- do not sort this array in this round
- include the two new arrays at the top level of `run_summary`

- [ ] **Step 5: Extend `run_summary.md` to render the new detail sections**

Minimal code target:

```python
lines.extend(
    [
        "",
        "## Ranked Model RSA Details",
        *(
            f"- {row['model_id']} | view={row['view_name']} | rsa={row['rsa_similarity']} | "
            f"p_raw={row.get('p_value_raw')} | p_fdr={row.get('p_value_fdr')} | "
            f"n={row.get('n_shared_entries')} | status={row.get('score_status')} | "
            f"top={row.get('is_top_model')}"
            for row in summary["ranked_model_rsa_details"]
        ) or ["- None"],
    ]
)
```

Implementation requirements:

- keep field order stable exactly as specified in the spec
- add the same style of section for `view_comparison_details`
- emit `- None` for empty arrays

- [ ] **Step 6: Remove dead plotting helpers if they become unused**

Implementation target:

- remove `_plot_ranked_model_rsa`
- remove `_plot_view_comparison_summary`

Requirement:

- do not leave unreachable dead plotting code behind if removal is straightforward

- [ ] **Step 7: Run the targeted tests to verify summary details now pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "ranked_model_rsa_details or view_comparison_details or markdown_summary_includes_detail_sections" -v`

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -k "writes_rsa_outputs or writes_rsa_prototype_context_outputs" -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "feat: move stage3 summary details into run summary"
```

## Task 4: Final Verification And Output Contract Check

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Run the focused RSA test modules**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

Run: `pixi run pytest -q`

Expected: PASS.

- [ ] **Step 3: Manually inspect the final output contract in a fresh Stage 3 output directory**

Checklist:

- `figures/single_stimulus_sensitivity.png` exists
- `figures/ranked_model_rsa.png` does not exist
- `figures/view_comparison_summary.png` does not exist
- `run_summary.json` contains `ranked_model_rsa_details`
- `run_summary.json` contains `view_comparison_details`
- `run_summary.md` contains both new detail sections

- [ ] **Step 4: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "test: verify stage3 summary artifact contract"
```

## Notes For Implementers

- Treat this as a contract change, not a compatibility layer project.
- Do not add aliases for `leave_one_stimulus_out_robustness`, `ranked_model_rsa`, or `view_comparison_summary`.
- Keep the summary arrays stable and explicit so downstream consumers can read them directly from JSON.
- If any external consumer depends on the removed figure artifacts, capture that as a follow-up compatibility task rather than reintroducing duplicate outputs here.

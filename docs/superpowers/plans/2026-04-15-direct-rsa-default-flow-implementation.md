# Direct RSA Default Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/run_rsa.py` run directly from preprocess outputs and `matrix.xlsx` by default for `global_profile`, while preserving curated `--model-input-root` as an advanced optional path.

**Architecture:** Reuse the existing Stage 3 model-resolution and RSA core instead of inventing a parallel path. Add a small direct-mode input synthesis layer in `model_space.py`, move the deterministic `stimulus -> sample_id` mapping logic into a shared home, and change `run_rsa.py` to branch between direct mode, curated mode, and the legacy geometry fallback based on explicit CLI precedence.

**Tech Stack:** Python, pandas, numpy, openpyxl, parquet IO, pytest, existing Stage 3 RSA modules

---

## File Map

- `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
  Responsibility: shared Stage 3 model-input validation and direct-mode input synthesis for `global_profile`.
- `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
  Responsibility: auto-seed builder support; updated only to reuse the shared mapping helper after it moves into `model_space.py`.
- `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
  Responsibility: CLI parsing, mode selection, and handoff into the existing RSA engine.
- `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`
  Responsibility: unit coverage for shared mapping rules and direct-mode input synthesis.
- `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
  Responsibility: builder compatibility coverage after shared mapping logic is reused.
- `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
  Responsibility: CLI behavior for direct mode, curated mode, and legacy geometry fallback.

### Task 1: Move Stimulus Mapping Logic Into Shared Model-Space Code

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`

- [ ] **Step 1: Write the failing shared-helper tests**

Add focused tests to `tests/test_model_space.py` covering the deterministic mapping contract currently living in the seed builder:

```python
def test_build_stimulus_sample_map_extracts_sample_id_from_stim_name():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
            {"stimulus": "b35_0", "stim_name": "A228 stationary"},
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
        ]
    )

    mapping = build_stimulus_sample_map(metadata, matrix_sample_ids=pd.Index(["A226", "A228"]))

    assert mapping.to_dict("records") == [
        {"stimulus": "b34_0", "stim_name": "A226 stationary", "sample_id": "A226"},
        {"stimulus": "b35_0", "stim_name": "A228 stationary", "sample_id": "A228"},
    ]


def test_build_stimulus_sample_map_rejects_conflicting_stim_name():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b34_0", "stim_name": "A226 stationary"},
            {"stimulus": "b34_0", "stim_name": "A227 stationary"},
        ]
    )

    with pytest.raises(ValueError, match="exactly one stim_name"):
        build_stimulus_sample_map(metadata, matrix_sample_ids=pd.Index(["A226", "A227"]))
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pixi run pytest tests/test_model_space.py -k "build_stimulus_sample_map" -q`

Expected: FAIL because `model_space.py` does not yet expose `build_stimulus_sample_map`.

- [ ] **Step 3: Move the mapping helper into `model_space.py` and reuse it from the seed builder**

Implement in `src/bacteria_analysis/model_space.py`:

```python
def build_stimulus_sample_map(metadata: pd.DataFrame, *, matrix_sample_ids: pd.Index) -> pd.DataFrame:
    ...


def _sample_id_from_stim_name(stim_name: str) -> str:
    ...
```

Then update `src/bacteria_analysis/model_space_seed.py` to import and reuse `build_stimulus_sample_map` instead of keeping a second copy.

- [ ] **Step 4: Run shared and builder compatibility tests**

Run: `pixi run pytest tests/test_model_space.py tests/test_model_space_seed.py -k "build_stimulus_sample_map" -q`

Expected: PASS.

- [ ] **Step 5: Commit the shared mapping refactor**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py
git commit -m "refactor: share Stage 3 stimulus mapping helper"
```

### Task 2: Add Direct-Mode Global-Profile Input Synthesis

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`

- [ ] **Step 1: Write failing tests for direct-mode input synthesis**

Add tests to `tests/test_model_space.py` for the new runtime resolver:

```python
def test_resolve_direct_global_profile_inputs_builds_minimal_stage3_bundle(tmp_path):
    matrix_path = _write_tiny_matrix_xlsx(tmp_path)
    preprocess_root = _write_tiny_preprocess_root(
        tmp_path,
        records=[
            {"stimulus": "b1_1", "stim_name": "A001 stationary"},
            {"stimulus": "b2_1", "stim_name": "A002 stationary"},
        ],
    )

    resolved = resolve_direct_global_profile_inputs(
        preprocess_root=preprocess_root,
        matrix_path=matrix_path,
    )

    assert resolved["model_registry_resolved"]["model_id"].tolist() == ["global_profile"]
    assert set(resolved["model_membership_resolved"]["metabolite_name"]) == {"feature_1", "feature_2"}
    assert resolved["stimulus_sample_map"]["sample_id"].tolist() == ["A001", "A002"]


def test_resolve_direct_global_profile_inputs_rejects_missing_matrix_sample_ids(tmp_path):
    ...
    with pytest.raises(ValueError, match="must exist in the matrix"):
        resolve_direct_global_profile_inputs(preprocess_root=preprocess_root, matrix_path=matrix_path)
```

- [ ] **Step 2: Run the direct-resolver tests to verify they fail**

Run: `pixi run pytest tests/test_model_space.py -k "resolve_direct_global_profile_inputs" -q`

Expected: FAIL because the direct-mode resolver does not exist yet.

- [ ] **Step 3: Implement the minimal direct-mode resolver in `model_space.py`**

Add a helper shaped like:

```python
def resolve_direct_global_profile_inputs(
    *,
    preprocess_root: str | Path,
    matrix_path: str | Path,
) -> dict[str, pd.DataFrame]:
    metadata = pd.read_parquet(Path(preprocess_root) / "trial_level" / "trial_metadata.parquet")
    matrix = read_metabolite_matrix(matrix_path)
    mapping = build_stimulus_sample_map(metadata, matrix_sample_ids=matrix.index)
    annotation = _build_minimal_metabolite_annotation(matrix)
    registry = _build_minimal_global_profile_registry()
    membership = _build_minimal_global_profile_membership(matrix)
    return _resolve_stage3_inputs(mapping, annotation, registry, membership, matrix_path)
```

Keep the synthesized annotation neutral and runtime-only; do not introduce any biochemical inference here.

- [ ] **Step 4: Run the model-space test slice**

Run: `pixi run pytest tests/test_model_space.py -k "resolve_direct_global_profile_inputs or global_profile" -q`

Expected: PASS.

- [ ] **Step 5: Commit the direct input synthesis layer**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py
git commit -m "feat: add direct global-profile Stage 3 inputs"
```

### Task 3: Change `run_rsa.py` to Make Direct Mode the Default

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Write failing CLI and path-resolution tests**

Add tests covering the new mode rules:

```python
def test_parse_args_defaults_model_input_root_to_none():
    args = RUN_RSA.parse_args(["--preprocess-root", "data/preprocess"])
    assert args.model_input_root is None


def test_cli_direct_mode_runs_without_model_input_root(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--preprocess-root",
            str(stage3_fixture_root / "preprocess"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--output-root",
            str(output_root),
            "--permutations",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cli_requires_preprocess_root_when_model_input_root_is_omitted(tmp_path, stage3_fixture_root):
    ...
    assert result.returncode == 1
    assert "preprocess-root" in result.stderr


def test_direct_mode_does_not_resolve_default_model_input_root(monkeypatch):
    def _unexpected(*args, **kwargs):
        raise AssertionError("resolve_model_input_root should not be called in direct mode")

    def _stop_direct_mode(**kwargs):
        raise RuntimeError("direct mode reached")

    monkeypatch.setattr(RUN_RSA, "resolve_model_input_root", _unexpected)
    monkeypatch.setattr(RUN_RSA, "resolve_direct_global_profile_inputs", _stop_direct_mode)

    with pytest.raises(RuntimeError, match="direct mode reached"):
        RUN_RSA.main(
            [
                "--preprocess-root",
                "data/preprocess",
                "--matrix",
                "data/matrix.xlsx",
                "--output-root",
                "results",
                "--permutations",
                "0",
            ]
        )
```

- [ ] **Step 2: Run the CLI-focused tests to verify they fail**

Run: `pixi run pytest tests/test_rsa_cli_smoke.py -k "direct_mode or defaults_model_input_root_to_none or requires_preprocess_root" -q`

Expected: FAIL because `run_rsa.py` still defaults `--model-input-root` to `data/model_space` and does not branch into direct mode.

- [ ] **Step 3: Implement the new CLI mode selection in `run_rsa.py`**

Make these exact changes:

- change `--model-input-root` default from `data/model_space` to `None`
- keep `--preprocess-root` default `None`
- keep `--geometry-root` for compatibility only
- in `main()`, branch on the raw parsed `args.model_input_root` value before calling any path resolver for model inputs
- only call `resolve_model_input_root(...)` inside branches where `args.model_input_root` was explicitly provided
- branch execution as:

```python
raw_model_input_root = args.model_input_root

if preprocess_root is not None and raw_model_input_root is None:
    resolved_inputs = resolve_direct_global_profile_inputs(
        preprocess_root=preprocess_root,
        matrix_path=matrix_path,
    )
    run_kwargs["aggregated_response_inputs"] = load_aggregated_response_context_inputs(...)
elif preprocess_root is not None and raw_model_input_root is not None:
    model_input_root = resolve_model_input_root(raw_model_input_root)
    resolved_inputs = resolve_model_inputs(model_input_root, matrix_path)
    run_kwargs["aggregated_response_inputs"] = load_aggregated_response_context_inputs(...)
elif preprocess_root is None and raw_model_input_root is not None:
    model_input_root = resolve_model_input_root(raw_model_input_root)
    resolved_inputs = resolve_model_inputs(model_input_root, matrix_path)
    run_kwargs["neural_matrices"] = load_geometry_pooled_neural_rdms(geometry_root)
else:
    raise ValueError("Direct RSA requires --preprocess-root when --model-input-root is omitted")
```

Update help text so the direct path is described as the primary workflow.

- [ ] **Step 4: Run the CLI smoke slice**

Run: `pixi run pytest tests/test_rsa_cli_smoke.py -k "direct_mode or writes_rsa_outputs or prototype_context_outputs" -q`

Expected: PASS.

- [ ] **Step 5: Commit the CLI refactor**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "feat: make direct RSA the default Stage 3 path"
```

### Task 4: Verify Curated Compatibility and Full Relevant Test Coverage

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Add one compatibility test for the legacy curated fallback**

Add a smoke test proving the old curated path still works when only `--geometry-root` and `--model-input-root` are supplied:

```python
def test_cli_legacy_curated_fallback_still_uses_geometry_root(tmp_path, stage3_fixture_root):
    ...
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_root / "rsa" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregated_response_context_enabled"] is False
```

- [ ] **Step 2: Run the full relevant test suite**

Run: `pixi run pytest tests/test_model_space.py tests/test_model_space_seed.py tests/test_rsa.py tests/test_rsa_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 3: Fix any fallout without widening scope**

Only address regressions caused by:

- moving the shared mapping helper
- adding the direct resolver
- changing the CLI precedence

Do not opportunistically refactor unrelated Stage 3 code in this task.

- [ ] **Step 4: Re-run the full relevant test suite**

Run: `pixi run pytest tests/test_model_space.py tests/test_model_space_seed.py tests/test_rsa.py tests/test_rsa_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the verification fixes**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "test: verify direct RSA and curated compatibility"
```

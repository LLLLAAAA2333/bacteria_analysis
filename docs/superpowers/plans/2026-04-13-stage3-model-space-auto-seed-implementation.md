# Stage 3 Model-Space Auto-Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Stage 3 model-space auto-seed pipeline that turns `matrix.xlsx` metabolite headers plus current trial metadata into provisional, evidence-backed `model_space` CSVs.

**Architecture:** Keep `scripts/run_rsa.py` and `src/bacteria_analysis/model_space.py` as consumers of finalized CSV inputs. Add a separate builder module and CLI that normalize matrix headers, resolve cached compound identities, enrich taxonomy evidence, apply registry-driven membership rules, write a review queue and manifest, and validate the generated output by reusing the existing `resolve_model_inputs` contract.

**Tech Stack:** Python 3.11, pandas, openpyxl, standard library (`argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`, `re`, `urllib`), pytest

---

## Overview

This plan implements the approved spec:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-04-13-stage3-model-space-auto-seed-design.md`

The implementation intentionally does not change:

- Stage 3 RSA scoring in `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
- model-space runtime resolution in `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- the `scripts/run_rsa.py` runtime boundary
- existing historical inputs under `H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603`

First generated real-data output should go to:

```text
H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603_auto/
```

## File Structure

- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
  Responsibility: deterministic header normalization, stimulus-to-sample map generation, cached identity/evidence parsing, taxonomy flattening, rule-based membership generation, manifest assembly, and output writing helpers.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py`
  Responsibility: thin CLI wrapper around `model_space_seed.build_model_space`; no RSA execution.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
  Responsibility: unit and integration tests for normalization, cached identity resolution, rule membership, review queue routing, manifest writing, and generated CSV validation.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/*.json`
  Responsibility: tiny cached PubChem-style fixtures used by tests; do not hit the network in tests.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/identity_resolution_evidence.csv`
  Responsibility: tiny identity evidence cache fixture used by the builder integration tests and CLI tests.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/taxonomy_enrichment.csv`
  Responsibility: tiny cache-style taxonomy fixture with ChEBI/HMDB/ClassyFire columns used to test enrichment without network or downloads.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`
  Responsibility: add an optional `model-space` task for the new builder CLI.
- Modify only if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
  Responsibility: expose existing canonicalization behavior through a small public helper only if duplicating it in `model_space_seed.py` would create drift. Do not add external lookup or builder logic here.

No dependency addition is required for the first implementation. Use `urllib.request` for optional live PubChem requests and keep ChEBI/HMDB as cache-driven enrichment in this slice.

The first slice does not implement live ChEBI or HMDB access. It accepts optional local identity and enrichment cache files. If identity evidence is absent and live PubChem refresh is not explicitly requested, the builder should emit schema-valid unresolved identity evidence for every normalized header and route those rows to review.

Header normalization must read raw Excel headers separately from `read_metabolite_matrix`, because the existing matrix loader canonicalizes metabolite names for Stage 3 runtime use. The builder should preserve both the raw header and the Stage 3 canonical metabolite name.

## Output Contract

The builder should write:

```text
model_space_202603_auto/
  stimulus_sample_map.csv
  metabolite_annotation.csv
  model_registry.csv
  model_membership.csv
  model_membership_review_queue.csv
  model_space_manifest.json
  cache/
    normalized_headers.csv
    identity_resolution_evidence.csv
    taxonomy_enrichment_evidence.csv
    membership_rule_evidence.csv
```

`model_membership.csv` should include only high-confidence automatic rows:

```text
membership_source = auto_taxonomy_rule
review_status = auto_high_confidence
ambiguous_flag = false
```

Low-confidence, unresolved, or multi-hit rows should go to `model_membership_review_queue.csv` instead of the default membership table.

Review queue schema:

```text
model_id,metabolite_name,membership_source,review_status,ambiguous_flag,reason,matched_rule,matched_field,matched_value,notes
```

For unresolved identities that do not match any model rule, use `model_id=""`, `membership_source="unresolved_identity"`, `review_status="unresolved"`, and `ambiguous_flag=true`.

## Task 1: Add Header Normalization Contracts

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`

- [ ] **Step 1: Write failing tests for deterministic metabolite header normalization**

Add tests covering:

- preserves `original_header`
- strips whitespace into `normalized_name`
- converts Greek-letter names and symbols such as beta variants to the ASCII token `beta`
- converts prime-like characters to ASCII apostrophe (`'`)
- extracts the final parenthetical alias into `alias_in_parentheses`
- emits `query_candidates` containing the full normalized name, the name without parenthetical alias, and the alias
- rejects duplicate `normalized_name` values
- rejects blank headers

Representative test:

```python
def test_normalize_metabolite_header_preserves_original_and_extracts_alias():
    record = normalize_metabolite_header(" Beta-Muricholic acid (beta-MCA) ")

    assert record["original_header"] == " Beta-Muricholic acid (beta-MCA) "
    assert record["normalized_name"] == "Beta-Muricholic acid (beta-MCA)"
    assert record["stage3_metabolite_name"] == "Beta-Muricholic acid (beta-MCA)"
    assert record["alias_in_parentheses"] == "beta-MCA"
    assert "Beta-Muricholic acid" in record["query_candidates"]
    assert "beta-MCA" in record["query_candidates"]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "normalize_metabolite_header or normalized_header_table" -v`

Expected: FAIL because `model_space_seed.py` does not exist.

- [ ] **Step 3: Implement minimal normalization helpers**

Add:

- `read_raw_metabolite_headers(matrix_path: str | Path) -> list[object]`
- `normalize_metabolite_header(header: object) -> dict[str, object]`
- `build_normalized_header_table(headers: list[object]) -> pd.DataFrame`
- private helpers for alias extraction and query-candidate construction

Implementation requirements:

- use `openpyxl.load_workbook(path, read_only=True, data_only=True)` to read the first worksheet's first row before `read_metabolite_matrix` canonicalizes headers
- skip the sample identifier column when returning raw metabolite headers
- keep the original header exactly
- normalize only deterministic local text issues
- return columns `original_header`, `normalized_name`, `stage3_metabolite_name`, `alias_in_parentheses`, `query_candidates`, `normalization_notes`
- set `stage3_metabolite_name` to the canonical column name that will appear in `read_metabolite_matrix`
- store `query_candidates` as a list in memory and as a stable delimiter-separated string only at CSV-write time
- reject duplicate normalized metabolite names

- [ ] **Step 4: Run targeted tests and verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "normalize_metabolite_header or normalized_header_table" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py
git commit -m "test: lock model-space header normalization"
```

## Task 2: Add Stimulus-To-Sample Map Generation

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`

- [ ] **Step 1: Write failing tests for sample ID extraction from trial metadata**

Add tests covering:

- `A226 stationary` maps to `sample_id=A226`
- repeated trials collapse to one row per `stimulus`
- one stimulus mapping to multiple `stim_name` values fails
- duplicate final `sample_id` values fail
- derived sample IDs absent from `matrix.xlsx` fail

Representative test:

```python
def test_build_stimulus_sample_map_extracts_sample_id_from_stationary_labels():
    metadata = pd.DataFrame.from_records([
        {"stimulus": "b34_0", "stim_name": "A226 stationary"},
        {"stimulus": "b35_0", "stim_name": "A228 stationary"},
        {"stimulus": "b34_0", "stim_name": "A226 stationary"},
    ])

    mapping = build_stimulus_sample_map(metadata, matrix_sample_ids=pd.Index(["A226", "A228"]))

    assert mapping.to_dict(orient="records") == [
        {"stimulus": "b34_0", "stim_name": "A226 stationary", "sample_id": "A226"},
        {"stimulus": "b35_0", "stim_name": "A228 stationary", "sample_id": "A228"},
    ]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "stimulus_sample_map" -v`

Expected: FAIL because `build_stimulus_sample_map` does not exist.

- [ ] **Step 3: Implement sample-map generation**

Add `build_stimulus_sample_map(metadata: pd.DataFrame, *, matrix_sample_ids: pd.Index) -> pd.DataFrame`.

Implementation requirements:

- require `stimulus` and `stim_name`
- derive `sample_id` from the first token of `stim_name`
- reject blanks
- reject one-to-many `stimulus -> stim_name`
- reject duplicate final `sample_id`
- validate all `sample_id` values exist in matrix index
- return columns in `("stimulus", "stim_name", "sample_id")`
- preserve deterministic first-occurrence ordering

- [ ] **Step 4: Run targeted tests and verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "stimulus_sample_map" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py
git commit -m "feat: derive model-space stimulus sample map"
```

## Task 3: Add Cached PubChem Identity Resolution

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/pubchem_cholic_acid.json`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/pubchem_multi_hit.json`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/identity_resolution_evidence.csv`

- [ ] **Step 1: Add small cached PubChem-style fixtures**

Create a high-confidence fixture with one candidate for `Cholic acid`, including `pubchem_cid`, `candidate_name`, `synonyms`, `smiles`, and `inchikey`.

Create an ambiguous fixture with two plausible candidates for the same query so the resolver can exercise `multi_hit_needs_review`.

The fixtures should be structurally representative, small, and deterministic. They do not need exhaustive chemistry.

Also create `identity_resolution_evidence.csv` with the same minimum columns required by `IDENTITY_EVIDENCE_COLUMNS`, including one high-confidence `Cholic acid (CA)` row and one unresolved row for builder integration tests.

In `tests/test_model_space_seed.py`, add a small fixture:

```python
@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "model_space_seed"
```

Minimum identity evidence schema:

```text
original_header,normalized_name,stage3_metabolite_name,canonical_compound_name,resolution_status,source,pubchem_cid,chebi_id,hmdb_id,inchikey,smiles,synonyms,match_score,match_reason,evidence_json
```

Resolved row shape:

```text
resolution_status=resolved_high_confidence
source=pubchem
pubchem_cid=<stable CID>
inchikey=<non-empty InChIKey>
match_score=<numeric score>
match_reason=<compact evidence string>
```

Unresolved row shape:

```text
resolution_status=unresolved
source=
pubchem_cid=
chebi_id=
hmdb_id=
inchikey=
match_score=0
match_reason=no_identity_candidates
evidence_json={}
```

- [ ] **Step 2: Write failing tests for candidate scoring and status assignment**

Add tests covering:

- exact candidate-name match with `InChIKey` returns `resolved_high_confidence`
- exact synonym match returns `resolved_high_confidence`
- zero candidates returns `unresolved`
- multiple equal plausible candidates returns `multi_hit_needs_review`
- low-scoring loose match returns `resolved_low_confidence`

Representative test:

```python
def test_resolve_identity_marks_exact_pubchem_match_high_confidence(fixture_path):
    record = {
        "original_header": "Cholic acid (CA)",
        "normalized_name": "Cholic acid (CA)",
        "alias_in_parentheses": "CA",
        "query_candidates": ["Cholic acid (CA)", "Cholic acid", "CA"],
    }
    payload = json.loads((fixture_path / "pubchem_cholic_acid.json").read_text(encoding="utf-8"))

    result = resolve_identity_from_cached_candidates(record, cached_payloads=[payload])

    assert result["resolution_status"] == "resolved_high_confidence"
    assert result["pubchem_cid"] == "221493"
    assert result["inchikey"]
    assert "exact" in result["match_reason"]
```

- [ ] **Step 3: Run the tests and verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "resolve_identity" -v`

Expected: FAIL because identity resolution helpers do not exist.

- [ ] **Step 4: Implement cached identity resolution helpers**

Add:

- `IDENTITY_EVIDENCE_COLUMNS`
- `load_identity_evidence_cache(path: str | Path | None) -> pd.DataFrame`
- `resolve_identity_from_cached_candidates(normalized_record: dict[str, object], *, cached_payloads: list[dict[str, object]]) -> dict[str, object]`
- `_collect_candidate_records(cached_payloads)`
- `_score_identity_candidate(normalized_record, candidate)`

Implementation requirements:

- keep one evidence output row per normalized metabolite
- return an empty DataFrame with `IDENTITY_EVIDENCE_COLUMNS` when no identity cache path is supplied
- validate required identity evidence cache columns when a path is supplied
- use exact normalized-name or synonym match as high-confidence evidence
- require `inchikey` for high-confidence identity when a source provides it
- mark zero candidates as `unresolved`
- mark multiple equal high-scoring candidates as `multi_hit_needs_review`
- keep `match_reason` and compact `evidence_json`
- do not call the network in this helper

- [ ] **Step 5: Add optional PubChem cache fetch helper behind an explicit flag**

Add `fetch_pubchem_payload(query: str, *, timeout_seconds: float = 20.0) -> dict[str, object]`.

Implementation requirements:

- use `urllib.request`
- build URLs against PubChem PUG-REST only
- return a normalized payload shape matching the fixture format
- do not call this helper unless the CLI receives an explicit `--refresh-pubchem-cache` flag
- keep tests mocked; do not introduce live-network tests

- [ ] **Step 6: Run targeted tests and verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "resolve_identity or pubchem" -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed
git commit -m "feat: add cached metabolite identity resolution"
```

## Task 4: Add Taxonomy Enrichment And Membership Rules

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/taxonomy_enrichment.csv`

- [ ] **Step 1: Write a failing test for taxonomy enrichment cache loading**

Create `taxonomy_enrichment.csv` with these minimum columns:

```text
normalized_name,pubchem_cid,chebi_id,hmdb_id,inchikey,chebi_parent_terms,hmdb_super_class,hmdb_class,hmdb_sub_class,classyfire_kingdom,classyfire_superclass,classyfire_class,classyfire_subclass,pathway_tags,annotation_confidence,annotation_status,annotation_evidence
```

Add a test like:

```python
def test_load_taxonomy_enrichment_cache_returns_schema_when_missing(tmp_path):
    enrichment = load_taxonomy_enrichment_cache(None)

    assert list(enrichment.columns) == TAXONOMY_ENRICHMENT_COLUMNS
    assert enrichment.empty


def test_load_taxonomy_enrichment_cache_reads_cached_chebi_hmdb_fields(fixture_path):
    enrichment = load_taxonomy_enrichment_cache(fixture_path / "taxonomy_enrichment.csv")

    row = enrichment.loc[enrichment["normalized_name"] == "Cholic acid (CA)"].iloc[0]
    assert row["chebi_parent_terms"] == "bile acid"
    assert row["hmdb_super_class"] == "Lipids and lipid-like molecules"
```

- [ ] **Step 2: Run the enrichment-cache tests and verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "taxonomy_enrichment_cache" -v`

Expected: FAIL because enrichment cache loading helpers do not exist.

- [ ] **Step 3: Implement taxonomy enrichment cache loading**

Add:

- `TAXONOMY_ENRICHMENT_COLUMNS`
- `load_taxonomy_enrichment_cache(path: str | Path | None) -> pd.DataFrame`
- `merge_identity_and_taxonomy_evidence(identity_evidence: pd.DataFrame, taxonomy_enrichment: pd.DataFrame) -> pd.DataFrame`

Implementation requirements:

- return an empty DataFrame with the required schema when no cache path is supplied
- validate required cache columns when a path is supplied
- join on `normalized_name`
- preserve identity fields when taxonomy fields are absent
- write the merged result to `cache/taxonomy_enrichment_evidence.csv` during builder orchestration
- do not call ChEBI, HMDB, or ClassyFire live in this task

- [ ] **Step 4: Write failing tests for taxonomy flattening**

Add tests covering:

- one annotation row is emitted per normalized matrix metabolite
- resolved high-confidence evidence maps to `review_status=auto_high_confidence`
- bile-acid taxonomy produces `pathway_tag=bile_acid`
- multi-hit or low-confidence evidence sets `ambiguous_flag=True`
- unresolved evidence stays in annotation but does not become model membership

Representative test:

```python
def test_build_metabolite_annotation_from_evidence_flattens_bile_acid_taxonomy():
    normalized = pd.DataFrame.from_records([
        {"normalized_name": "Cholic acid (CA)", "original_header": "Cholic acid (CA)"}
    ])
    evidence = pd.DataFrame.from_records([
        {
            "normalized_name": "Cholic acid (CA)",
            "canonical_compound_name": "Cholic acid",
            "resolution_status": "resolved_high_confidence",
            "chebi_parent_terms": "bile acid",
            "hmdb_super_class": "Lipids and lipid-like molecules",
            "classyfire_subclass": "Bile acids, alcohols and derivatives",
        }
    ])

    annotation = build_metabolite_annotation_from_evidence(normalized, evidence)

    assert annotation.loc[0, "metabolite_name"] == "Cholic acid (CA)"
    assert annotation.loc[0, "superclass"] == "Lipids and lipid-like molecules"
    assert annotation.loc[0, "pathway_tag"] == "bile_acid"
    assert annotation.loc[0, "review_status"] == "auto_high_confidence"
```

- [ ] **Step 5: Write failing tests for membership rule generation across registry rows**

Add tests covering:

- `bile_acid` auto-high-confidence annotations enter `model_membership.csv`
- `global_profile` membership is not generated because existing resolver seeds it from all matrix columns
- low-confidence matches enter review queue, not membership
- unresolved metabolites enter review queue with `review_status=unresolved`
- excluded registry rows are ignored
- output has unique `(model_id, metabolite_name)` pairs

Representative test:

```python
def test_build_model_membership_from_rules_adds_high_confidence_bile_acid_membership():
    registry = _registry_frame(["global_profile", "bile_acid"])
    annotation = pd.DataFrame.from_records([
        {
            "metabolite_name": "Cholic acid (CA)",
            "superclass": "Lipids and lipid-like molecules",
            "subclass": "Bile acids, alcohols and derivatives",
            "pathway_tag": "bile_acid",
            "annotation_source": "auto_identity_resolution",
            "review_status": "auto_high_confidence",
            "ambiguous_flag": False,
            "notes": "rule=bile_acid",
        }
    ])

    membership, review_queue, evidence = build_model_membership_from_rules(annotation, registry)

    assert membership[["model_id", "metabolite_name", "review_status"]].to_dict(orient="records") == [
        {"model_id": "bile_acid", "metabolite_name": "Cholic acid (CA)", "review_status": "auto_high_confidence"}
    ]
    assert review_queue.empty
    assert not evidence.empty
```

- [ ] **Step 6: Run the tests and verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "taxonomy or membership_from_rules" -v`

Expected: FAIL because taxonomy and rule helpers do not exist.

- [ ] **Step 7: Implement annotation flattening**

Add `build_metabolite_annotation_from_evidence(normalized_headers: pd.DataFrame, identity_evidence: pd.DataFrame) -> pd.DataFrame`.

Implementation requirements:

- output exactly the existing Stage 3 annotation columns
- preserve one row per normalized matrix metabolite
- set `metabolite_name` from `stage3_metabolite_name`, not raw `original_header`
- set `review_status` to `auto_high_confidence`, `needs_review`, or `unresolved`
- set `ambiguous_flag=True` for multi-hit or low-confidence evidence
- keep richer evidence in cache files, not in the simplified Stage 3 CSV

- [ ] **Step 8: Implement conservative membership rules**

Add `build_model_membership_from_rules(annotation: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`.

Initial rules:

```python
MODEL_RULE_KEYWORDS = {
    "bile_acid": ("bile acid", "bile acids", "cholic acid", "muricholic acid"),
    "indole_tryptophan": ("indole", "tryptophan", "kynurenine"),
    "phenyl_phenol": ("phenyl", "phenol", "phenolic", "benzoic acid"),
    "tca_organic_acid": ("citric acid cycle", "tricarboxylic acid", "succinate", "fumarate", "malate", "citrate"),
    "nucleotide_energy": ("nucleotide", "adenosine", "guanosine", "atp", "adp", "amp"),
    "broad_aromatic_union": ("aromatic", "indole", "phenyl", "phenol", "benzoic"),
}
```

Implementation requirements:

- apply rules only to registry rows whose `model_status != "excluded"`
- include only annotation rows with `review_status == "auto_high_confidence"` in default membership
- route lower-confidence matches to review queue
- route unresolved annotations to review queue even when no model rule matched, using `model_id=""`, `membership_source="unresolved_identity"`, `review_status="unresolved"`, and `ambiguous_flag=True`
- never generate `global_profile` membership
- keep one unique `(model_id, metabolite_name)` pair
- keep `notes` compact as `rule=<model_id>`
- emit rule evidence rows with `model_id`, `metabolite_name`, `matched_field`, `matched_value`, `rule_name`, and `decision`

- [ ] **Step 9: Run targeted tests and verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "taxonomy or membership_from_rules" -v`

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/fixtures/model_space_seed/taxonomy_enrichment.csv
git commit -m "feat: seed model memberships from taxonomy rules"
```

## Task 5: Add Builder Orchestration And CLI

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`

- [ ] **Step 1: Write failing integration test for generated output files**

Create a tiny `matrix.xlsx`, `trial_metadata.parquet`, registry CSV, and identity evidence fixture, then assert the builder writes all required CSVs, cache tables, and manifest.

Representative test:

```python
def test_build_model_space_writes_csvs_review_queue_cache_and_manifest(tmp_path, fixture_path):
    matrix_path, preprocess_root, registry_path = _write_tiny_builder_inputs(tmp_path)
    output_root = tmp_path / "model_space_auto"

    result = build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=fixture_path / "identity_resolution_evidence.csv",
        taxonomy_enrichment_path=fixture_path / "taxonomy_enrichment.csv",
        cache_version="test-cache-v1",
        refresh_pubchem_cache=False,
    )

    assert (output_root / "stimulus_sample_map.csv").exists()
    assert (output_root / "metabolite_annotation.csv").exists()
    assert (output_root / "model_registry.csv").exists()
    assert (output_root / "model_membership.csv").exists()
    assert (output_root / "model_membership_review_queue.csv").exists()
    assert (output_root / "model_space_manifest.json").exists()
    assert (output_root / "cache" / "normalized_headers.csv").exists()
    assert result["output_root"] == output_root
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "build_model_space_writes" -v`

Expected: FAIL because `build_model_space` does not exist.

- [ ] **Step 3: Implement `build_model_space` orchestration**

Add:

Function signature: `build_model_space(*, matrix_path: str | Path, preprocess_root: str | Path, registry_path: str | Path, output_root: str | Path, identity_evidence_path: str | Path | None = None, taxonomy_enrichment_path: str | Path | None = None, cache_version: str = "manual-cache-v1", refresh_pubchem_cache: bool = False) -> dict[str, Path]`

Implementation requirements:

- read matrix through existing `read_metabolite_matrix`
- read raw metabolite headers through `read_raw_metabolite_headers(matrix_path)` before using canonicalized matrix columns
- verify the count of raw metabolite headers matches the count of canonical matrix columns
- read trial metadata from `preprocess_root / "trial_level" / "trial_metadata.parquet"`
- read registry through existing `load_model_registry`
- call normalization, sample-map generation, annotation building, membership rule generation
- load optional identity evidence through `load_identity_evidence_cache(identity_evidence_path)`
- when identity evidence cache is absent and `refresh_pubchem_cache=False`, create unresolved identity evidence rows for every normalized header
- load optional taxonomy enrichment through `load_taxonomy_enrichment_cache(taxonomy_enrichment_path)`
- merge identity evidence and taxonomy enrichment before building `metabolite_annotation.csv`
- write `model_registry.csv` in normalized validated form
- write all CSVs with `index=False`
- create `cache/` before writing cache outputs
- validate generated CSVs by calling `resolve_model_inputs(output_root, matrix_path)`
- return a dictionary of written paths
- do not call live PubChem by default

- [ ] **Step 4: Implement manifest creation**

Add `build_model_space_manifest` with explicit matrix, metadata, registry, rule, and output metadata fields.

Manifest requirements:

- include `matrix_path` and matrix file SHA256
- include `preprocess_root`, trial metadata path, and trial metadata file SHA256
- include `registry_path` and registry file SHA256
- include identity evidence cache path and file SHA256 when supplied
- include `rule_version`
- include `cache_version`
- include generation timestamp
- include source policy fields: `pubchem_mode`, `chebi_mode`, `hmdb_mode`, `classyfire_mode`
- include taxonomy enrichment cache path and file SHA256 when supplied
- include output filenames

Use existing `bacteria_analysis.io.write_json` if helpful.

- [ ] **Step 5: Add CLI wrapper**

Create `scripts/build_model_space.py` with:

```python
"""CLI entry point for Stage 3 model-space auto-seeding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.model_space_seed import build_model_space


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provisional Stage 3 model-space CSV inputs.")
    parser.add_argument("--matrix", default="data/matrix.xlsx")
    parser.add_argument("--preprocess-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--identity-evidence-cache", default=None)
    parser.add_argument("--taxonomy-enrichment-cache", default=None)
    parser.add_argument("--cache-version", default="manual-cache-v1")
    parser.add_argument("--refresh-pubchem-cache", action="store_true")
    return parser.parse_args(argv)
```

Implementation requirements:

- `main(argv: list[str] | None = None)` should call `build_model_space`
- pass `identity_evidence_path=args.identity_evidence_cache`
- pass `taxonomy_enrichment_path=args.taxonomy_enrichment_cache`
- pass `cache_version=args.cache_version`
- return `1` with a concise stderr message on failure
- print the output root on success
- do not import or call Stage 3 RSA

- [ ] **Step 6: Add pixi task**

Modify `pixi.toml`:

```toml
model-space = "python scripts/build_model_space.py"
```

Keep existing tasks unchanged.

- [ ] **Step 7: Run targeted tests and CLI smoke test**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "build_model_space_writes or manifest or build_model_space_cli" -v`

Expected: PASS after adding the CLI test.

- [ ] **Step 8: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/pixi.toml
git commit -m "feat: add stage3 model-space builder"
```

## Task 6: Validate Generated Inputs Against Existing Stage 3 Contracts

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`

- [ ] **Step 1: Write failing test that generated output is accepted by `resolve_model_inputs`**

Add:

```python
def test_generated_model_space_resolves_with_existing_stage3_loader(tmp_path):
    matrix_path, preprocess_root, registry_path = _write_tiny_builder_inputs(tmp_path)
    output_root = tmp_path / "model_space_auto"

    build_model_space(
        matrix_path=matrix_path,
        preprocess_root=preprocess_root,
        registry_path=registry_path,
        output_root=output_root,
        identity_evidence_path=_tiny_identity_evidence_path(tmp_path),
        taxonomy_enrichment_path=_tiny_taxonomy_enrichment_path(tmp_path),
        cache_version="test-cache-v1",
        refresh_pubchem_cache=False,
    )

    resolved = resolve_model_inputs(output_root, matrix_path)
    non_global_membership = resolved["model_membership_resolved"].loc[
        resolved["model_membership_resolved"]["model_id"].astype(str) != "global_profile"
    ]

    assert not resolved["stimulus_sample_map"].empty
    assert "global_profile" in set(resolved["model_registry_resolved"]["model_id"])
    assert set(non_global_membership["review_status"]) <= {"auto_high_confidence"}
```

- [ ] **Step 2: Run the test and verify it fails if any schema drift exists**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -k "generated_model_space_resolves" -v`

Expected: initially FAIL only if Task 5 did not already validate through `resolve_model_inputs`; otherwise PASS.

- [ ] **Step 3: Fix schema drift only in the builder**

Requirements:

- do not loosen `model_space.py` validation to accommodate bad generated output
- make generated CSVs match the existing `model_space.py` schema
- preserve only existing optional membership columns in the final membership CSV
- keep expanded evidence in cache files

- [ ] **Step 4: Run all model-space tests**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py
git commit -m "test: validate generated model-space contract"
```

## Task 7: Real-Data Dry Run Without Overwriting Historical Inputs

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py`
- Do not modify: `H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603`

- [ ] **Step 1: Run builder against the 202603 dataset into a new output directory**

Run:

```powershell
pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --preprocess-root H:/Process_temporary/WJH/bacteria_analysis/data/20260313_preprocess --registry H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603/model_registry.csv --output-root H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603_auto --cache-version 20260313-initial-auto-seed-v1
```

Expected:

- exit code `0`
- new directory `data/20260313/model_space_202603_auto/`
- `stimulus_sample_map.csv` maps the 202603 trial metadata panel
- `model_membership.csv` contains only `auto_high_confidence` rows if identity evidence is available
- low-confidence or unresolved candidates appear in `model_membership_review_queue.csv`
- `model_space_manifest.json` exists
- manifest includes `cache_version=20260313-initial-auto-seed-v1`

If identity evidence is not yet cached for real data, it is acceptable for this first dry run to generate sparse membership and a large review queue. If a local identity evidence cache exists before the dry run, add `--identity-evidence-cache <path-to-cache.csv>` so the run is reproducible without live PubChem calls.

If a reviewed local enrichment cache exists before the dry run, add `--taxonomy-enrichment-cache <path-to-cache.csv>`. If it does not exist, the builder should still write schema-valid empty taxonomy evidence.

- [ ] **Step 2: Validate the generated directory through Stage 3 resolver**

Run:

```powershell
pixi run python -c "from bacteria_analysis.model_space import resolve_model_inputs; resolve_model_inputs(r'H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603_auto', r'H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx'); print('resolved')"
```

Expected: prints `resolved`.

- [ ] **Step 3: Run Stage 3 with zero permutations against the generated model space**

Run:

```powershell
pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py --stage2-root H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage2_geometry --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --model-input-root H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603_auto --preprocess-root H:/Process_temporary/WJH/bacteria_analysis/data/20260313_preprocess --permutations 0 --output-root H:/Process_temporary/WJH/bacteria_analysis/.tmp_stage3_model_space_auto
```

Expected:

- exit code `0`
- Stage 3 outputs written under `.tmp_stage3_model_space_auto/rsa`
- run summary labels models as provisional if auto-seeded membership is included

- [ ] **Step 4: Inspect generated summaries**

Inspect:

- `H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603_auto/model_space_manifest.json`
- `H:/Process_temporary/WJH/bacteria_analysis/data/20260313/model_space_202603_auto/model_membership_review_queue.csv`
- `H:/Process_temporary/WJH/bacteria_analysis/.tmp_stage3_model_space_auto/rsa/run_summary.json`

Expected:

- manifest points to the correct input matrix and trial metadata
- review queue is present even if empty
- Stage 3 output does not claim manual curation for auto-seeded rows

- [ ] **Step 5: Commit only source/test/task changes, not generated real-data outputs unless explicitly approved**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/pixi.toml
git commit -m "feat: verify stage3 model-space auto-seed builder"
```

Do not commit `data/20260313/model_space_202603_auto/` unless the user explicitly decides generated data should be versioned.

## Task 8: Final Verification

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`

- [ ] **Step 1: Run focused model-space tests**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py -q`

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `pixi run pytest -q`

Expected: PASS.

- [ ] **Step 3: Review generated plan/design alignment**

Checklist:

- builder is separate from `run_rsa.py`
- no live network in tests
- PubChem live refresh requires explicit flag
- ChEBI/HMDB are cache-driven or fixture-driven in this slice
- `auto_high_confidence` appears in membership, not `manually_curated`
- low-confidence and multi-hit rows go to review queue
- generated output validates through `resolve_model_inputs`

- [ ] **Step 4: Commit final fixes if any**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/scripts/build_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space_seed.py H:/Process_temporary/WJH/bacteria_analysis/pixi.toml
git commit -m "test: finalize stage3 model-space auto-seed workflow"
```

## Notes For Implementers

- Treat this as a builder project, not a Stage 3 RSA scoring change.
- Do not add `requests`, `bioservices`, ontology frameworks, or other new dependencies in the first slice.
- Do not let external API calls run during tests.
- Keep rich identity and taxonomy evidence out of the simplified Stage 3 CSVs and in cache/evidence files.
- Treat real-data generated outputs as artifacts requiring explicit user approval before committing.
- If the user later wants full HMDB batch enrichment, implement it as a separate task using downloaded HMDB data, not page scraping.
- If the user later wants full ChEBI ontology traversal, implement it as a separate task against ChEBI/OLS cache data rather than embedding that complexity in the first builder.

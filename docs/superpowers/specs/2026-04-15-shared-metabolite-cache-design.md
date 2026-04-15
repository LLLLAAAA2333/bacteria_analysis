# Shared Metabolite Cache Design

Date: 2026-04-15
Status: Draft for user review
Topic: Build a project-level reusable metabolite identity and taxonomy cache for Stage 3 model-space generation

## Goal

The current Stage 3 builder can consume identity and taxonomy cache files, but it does not generate a reusable project-level cache. It also only supports live refresh for PubChem. The goal of this design is to add a separate shared-cache builder that:

1. reads the stable metabolite panel from `matrix.xlsx`,
2. reuses previously curated identity and taxonomy evidence when available,
3. enriches missing or unresolved rows through a mixed source policy,
4. writes canonical cache files that can be reused across `202603` and future batches without rebuilding per-batch chemistry knowledge,
5. keeps Stage 3 `model-space` generation as a downstream consumer rather than turning it into a networked enrichment pipeline.

The result should be a project asset, not a batch-specific artifact.

## Non-Goals

- Do not change `scripts/run_rsa.py` runtime behavior.
- Do not make `pixi run model-space` query external services directly for ChEBI, HMDB, or ClassyFire.
- Do not scrape HMDB or ClassyFire web pages.
- Do not introduce a large ETL framework or new heavy dependencies.
- Do not treat auto-generated cache rows as manually curated scientific truth.

## Current Constraints

The existing Stage 3 builder already defines the canonical cache schemas through:

- `IDENTITY_EVIDENCE_COLUMNS`
- `TAXONOMY_ENRICHMENT_COLUMNS`

Those schemas are already accepted by:

- `load_identity_evidence_cache(...)`
- `load_taxonomy_enrichment_cache(...)`
- `build_model_space(...)`

This design keeps those schemas as the contract so the new shared cache can plug directly into the existing Stage 3 builder without changing downstream runtime code.

The current implementation only supports live PubChem refresh. ChEBI, HMDB, and ClassyFire are still cache-only in the Stage 3 builder.

For this design, the new shared-cache builder becomes the only component that is allowed to perform routine live PubChem or ChEBI enrichment. `build_model_space(...)` remains a downstream consumer of prebuilt cache files.

If the repository temporarily retains `--refresh-pubchem-cache` in `build_model_space(...)` for backward compatibility, treat that as a legacy compatibility path rather than the recommended or expanded architecture for this feature.

## Recommended Architecture

Add a separate shared-cache builder instead of expanding `pixi run model-space`.

Recommended flow:

```text
matrix.xlsx
-> normalized metabolite header table
-> merge with existing shared cache
-> refresh missing identity rows from PubChem
-> enrich taxonomy from ChEBI online lookups
-> merge optional local HMDB import
-> merge optional local ClassyFire import
-> conflict review queue
-> canonical shared cache outputs
-> pixi run model-space consumes the canonical shared cache
```

Recommended new files:

- `src/bacteria_analysis/metabolite_cache.py`
- `scripts/build_metabolite_cache.py`
- `tests/test_metabolite_cache.py`
- small fixtures under `tests/fixtures/metabolite_cache/`

Recommended reusable cache root:

```text
data/shared_caches/model_space/
  identity_resolution_evidence.csv
  taxonomy_enrichment_evidence.csv
  metabolite_cache_manifest.json
  metabolite_cache_review_queue.csv
  cache/
    normalized_headers.csv
    pubchem_candidate_cache.jsonl
    chebi_lookup_cache.jsonl
    hmdb_import_snapshot.csv
    classyfire_import_snapshot.csv
```

The canonical outputs at the root are the long-lived project assets. The `cache/` directory stores reproducibility evidence and source snapshots.

## Source Policy

Use a mixed source policy:

### Online sources

- `PubChem`: identity lookup only
- `ChEBI`: taxonomy and ontology enrichment only

### Local sources

- `HMDB`: local CSV or TSV import
- `ClassyFire`: local CSV or TSV import

### Why this split

- PubChem and ChEBI have stable official programmatic access paths that are suitable for targeted lookup.
- HMDB and ClassyFire are better treated as local imports in the first slice because they are higher-friction sources and should not turn routine cache refresh into a brittle network workflow.

## Shared Cache Contracts

The shared cache builder should write the same canonical contracts already used by Stage 3.

### Identity cache

`identity_resolution_evidence.csv` must follow `IDENTITY_EVIDENCE_COLUMNS` exactly, including:

- `original_header`
- `normalized_name`
- `stage3_metabolite_name`
- `canonical_compound_name`
- `resolution_status`
- `source`
- `pubchem_cid`
- `chebi_id`
- `hmdb_id`
- `inchikey`
- `smiles`
- `synonyms`
- `match_score`
- `match_reason`
- `evidence_json`

The shared cache must remain byte-compatible with the current Stage 3 identity-cache loader contract, not just conceptually similar.

### Taxonomy cache

`taxonomy_enrichment_evidence.csv` must follow `TAXONOMY_ENRICHMENT_COLUMNS` exactly, including:

- `normalized_name`
- `pubchem_cid`
- `chebi_id`
- `hmdb_id`
- `inchikey`
- `chebi_parent_terms`
- `hmdb_super_class`
- `hmdb_class`
- `hmdb_sub_class`
- `classyfire_kingdom`
- `classyfire_superclass`
- `classyfire_class`
- `classyfire_subclass`
- `pathway_tags`
- `annotation_confidence`
- `annotation_status`
- `annotation_evidence`

The builder must never invent a second schema for the same concepts.

## Identity Resolution Policy

Identity resolution remains conservative.

### Default behavior

- Reuse an existing shared-cache row when `normalized_name` already exists and the row is not unresolved.
- Reuse a previously curated row even if a live source now returns a different answer.
- Only call PubChem for rows that are missing, unresolved, or explicitly forced to refresh.

### Confidence policy

- `resolved_high_confidence`: exact normalized-name or synonym match plus stable structural identity when available
- `resolved_low_confidence`: plausible but weaker candidate
- `multi_hit_needs_review`: multiple equal top candidates
- `unresolved`: no acceptable candidate

### Review safety

Live refresh must not silently overwrite a reviewed row with conflicting `inchikey`, `pubchem_cid`, `chebi_id`, or `hmdb_id`.

Conflicting rows go to the review queue instead.

## Taxonomy Enrichment Policy

Taxonomy enrichment should layer sources rather than choose only one.

Recommended precedence for each taxonomy field:

1. reviewed existing shared cache
2. local HMDB import
3. local ClassyFire import
4. live ChEBI lookup
5. blank value

Rationale:

- Existing reviewed cache is the most trustworthy project truth.
- HMDB and ClassyFire provide broad taxonomy that should not be discarded if already curated locally.
- ChEBI is strong for ontology and parent-term enrichment but should not erase stronger reviewed project data.

Recommended field ownership:

- `chebi_id`, `chebi_parent_terms`: primarily ChEBI
- `hmdb_id`, `hmdb_super_class`, `hmdb_class`, `hmdb_sub_class`: primarily HMDB
- `classyfire_*`: primarily ClassyFire
- `pathway_tags`: derived from the merged taxonomy plus existing local pathway heuristics

## Local Import Contracts

The first implementation should support normalized local imports instead of trying to parse every official upstream dump format.

Recommended CLI inputs:

- `--hmdb-import <path>`
- `--classyfire-import <path>`

Recommended accepted columns for HMDB import:

- `normalized_name`
- `hmdb_id`
- `inchikey`
- `super_class`
- `class`
- `sub_class`
- optional evidence/status columns

Recommended accepted columns for ClassyFire import:

- `normalized_name`
- `inchikey`
- `kingdom`
- `superclass`
- `class`
- `subclass`
- optional evidence/status columns

The builder should normalize these inputs into the canonical taxonomy cache schema and reject duplicate `normalized_name` rows.

The first slice does not need to auto-parse raw HMDB XML or arbitrary ClassyFire JSON dumps.

## Conflict Policy

The builder should fail safely at the row level, not at the whole-run level, when sources disagree.

### Hard conflict signals

- mismatched `inchikey` for the same `normalized_name`
- mismatched stable external IDs for a row that is already reviewed

### Conflict handling

- preserve the existing canonical row unchanged
- append a row to `metabolite_cache_review_queue.csv`
- record the conflicting source payload or compact evidence in `annotation_evidence` or `evidence_json`
- continue processing other metabolites

### Review queue categories

Use compact reasons such as:

- `identity_conflict`
- `taxonomy_conflict`
- `multi_hit_needs_review`
- `missing_taxonomy`
- `missing_identity`

### Review queue schema

`metabolite_cache_review_queue.csv` should be a flat one-row-per-issue table with:

- `normalized_name`
- `stage3_metabolite_name`
- `issue_type`
- `source_name`
- `field_name`
- `existing_value`
- `incoming_value`
- `status`
- `notes`

Rules:

- write one row per `(normalized_name, issue_type, source_name, field_name)` issue
- use `field_name=""` when the issue is row-level rather than field-level
- use `status` values such as `needs_review` and `unresolved`
- do not silently collapse multiple independent source conflicts into one opaque row

## Refresh Policy

The shared cache builder should be incremental by default.

### Default run

- load the existing shared cache if present
- only enrich rows that are missing or unresolved
- preserve previously reviewed rows

### Explicit refresh flags

Recommended flags:

- `--refresh-pubchem`
- `--refresh-chebi`

In the first implementation, these flags should still refresh only rows that are missing or unresolved. They are source-enable flags, not broad rebuild flags.

If a future run needs a full rebuild of already resolved rows, that should be a separate explicit task rather than a hidden side effect of the first implementation.

## CLI Contract

Recommended command:

```text
pixi run metabolite-cache --matrix data/matrix.xlsx --output-root data/shared_caches/model_space
```

Recommended additional arguments:

- `--existing-identity-cache`
- `--existing-taxonomy-cache`
- `--hmdb-import`
- `--classyfire-import`
- `--cache-version`
- `--refresh-pubchem`
- `--refresh-chebi`

Success behavior:

- writes canonical cache files
- writes raw/source evidence snapshots under `cache/`
- prints the output root

Failure behavior:

- returns non-zero with concise stderr on schema or file errors
- does not partially overwrite canonical outputs if the write would be structurally invalid

## Downstream Usage

After the shared cache exists, Stage 3 generation should become:

```text
pixi run metabolite-cache ...
pixi run model-space --identity-evidence-cache data/shared_caches/model_space/identity_resolution_evidence.csv --taxonomy-enrichment-cache data/shared_caches/model_space/taxonomy_enrichment_evidence.csv ...
```

This keeps chemistry knowledge project-level and keeps per-batch outputs derived.

The recommended architecture is:

- `pixi run metabolite-cache` may call live PubChem and live ChEBI
- `pixi run model-space` consumes only local cache files
- `pixi run python scripts/run_rsa.py ...` consumes only local model-space outputs

The new implementation should not add any new live-network responsibilities to `build_model_space(...)` or `run_rsa.py`.

Reusable:

- `identity_resolution_evidence.csv`
- `taxonomy_enrichment_evidence.csv`

Batch-specific derived outputs:

- `stimulus_sample_map.csv`
- `metabolite_annotation.csv`
- `model_membership.csv`
- `model_membership_review_queue.csv`

## Manifest And Reproducibility

The shared cache builder should write `metabolite_cache_manifest.json` with:

- `matrix_path`
- matrix SHA256
- source file paths and hashes for local HMDB and ClassyFire imports when provided
- flags indicating whether PubChem and ChEBI live refresh were enabled
- cache version
- generation timestamp
- output filenames

This manifest is for provenance, not cache invalidation logic.

## Error Handling

Fail fast when:

- `matrix.xlsx` contains blank or duplicate normalized metabolite names
- a local import misses required columns
- a local import has duplicate `normalized_name` rows
- a canonical output row violates the required schema

Do not fail the entire run when:

- one metabolite is unresolved
- one metabolite has conflicting enrichment
- one online lookup misses

Those cases should route to the review queue.

## Output Ordering

Canonical shared-cache outputs should use deterministic ordering so repeated runs produce stable diffs.

Recommended ordering:

- root canonical CSVs sorted by `normalized_name` ascending
- review queue sorted by `normalized_name`, then `issue_type`, then `source_name`, then `field_name`
- evidence snapshot files preserve source-native ordering only when that ordering is already deterministic; otherwise sort by the primary lookup key before writing

## Testing Strategy

Add deterministic tests for:

- incremental reuse of an existing shared cache
- PubChem refresh only touching missing or unresolved rows
- ChEBI enrichment only filling the fields it owns
- HMDB and ClassyFire import normalization
- conflict routing without silent overwrite
- generated canonical cache compatibility with `build_model_space(...)`

All online lookups must be mocked in tests.

## Recommended First Implementation Slice

The first implementation should include:

- a new `metabolite_cache.py` module
- a `build_metabolite_cache.py` CLI
- a `pixi` task named `metabolite-cache`
- online PubChem identity refresh
- online ChEBI enrichment
- local HMDB import normalization
- local ClassyFire import normalization
- shared cache review queue
- manifest output
- tests proving the outputs are directly consumable by the existing `model-space` builder

This slice is enough to produce a reusable project-level cache without expanding Stage 3 runtime responsibilities.

# Stage 3 Model-Space Auto-Seed Design

Date: 2026-04-13
Status: Draft for user review
Topic: Make Stage 3 biochemical model-space inputs reproducible while preserving review boundaries

> Update 2026-04-18: local verification showed that
> `data/metabolism_raw_data.xlsx` already contains metabolite identity and
> taxonomy fields aligned to `data/matrix.xlsx` on a 380-to-380 panel after
> small local name normalization. For the current project data, external
> identity resolution should be treated as a legacy fallback for missing
> metadata rather than the default design path.

## Goal

Stage 3 currently consumes a `model_space` directory containing four CSV files:

- `stimulus_sample_map.csv`
- `metabolite_annotation.csv`
- `model_registry.csv`
- `model_membership.csv`

Those files are valid scientific inputs, but the current `202603` version was produced by a one-off Codex seeding pass. The goal of this design is to make that seeding reproducible:

1. derive stable metabolite identities from `matrix.xlsx` headers,
2. enrich them with external chemical and metabolite classifications,
3. translate those annotations into draft model memberships for all existing registry families,
4. preserve evidence and confidence so the draft model space can be audited later.

This is not a replacement for manual curation. It is an automation layer that reduces manual effort and marks unresolved or ambiguous entries for later review.

## Non-Goals

- Do not change the core Stage 3 RSA scoring logic.
- Do not make external database results equivalent to manual curation.
- Do not silently generate or modify `model_space` inputs during `scripts/run_rsa.py`.
- Do not scrape web pages when an official API or downloadable dataset is available.
- Do not require detailed manual review before the project can run exploratory Stage 3 models.

## Current Input Semantics

`matrix.xlsx` is the upstream numeric feature source. Its rows are metabolite samples and its columns are metabolite feature names. Stage 3 models are built by selecting rows and columns from this matrix, preprocessing the resulting feature matrix, computing a model RDM, and comparing that model RDM to a neural RDM.

`stimulus_sample_map.csv` maps neural stimuli to matrix rows. It should be generated from the current preprocessing output where possible, then validated against `matrix.xlsx`.

`metabolite_annotation.csv` describes matrix columns. It should retain one row per matrix metabolite and hold normalized identity, external IDs, taxonomy, ontology, pathway tags, status, and notes.

`model_registry.csv` defines the candidate models. It is a small human-maintained table that names each model, its tier, its status, its feature kind, and its distance rule.

`model_membership.csv` maps models to metabolites. It should become a generated draft table from annotation and rules, with manual overrides allowed.

## Recommended Pipeline

The pipeline should be explicit and staged:

```text
matrix.xlsx headers
-> local name normalization
-> compound identity resolution
-> taxonomy and ontology enrichment
-> rule-based model membership seeding
-> review queue and manual overrides
-> final model_space CSVs
```

Each stage should write or cache enough evidence to reproduce why a metabolite was assigned to a model.

## Stage 1: Local Name Normalization

This stage is deterministic and local. It does not call external databases.

Inputs:

- raw metabolite headers from `matrix.xlsx`

Outputs:

- `original_header`
- `normalized_name`
- `alias_in_parentheses`
- `query_candidates`
- `normalization_notes`

Responsibilities:

- strip leading and trailing whitespace,
- normalize known Greek-letter spellings and symbols,
- normalize prime characters,
- preserve parenthetical aliases such as `CA` or `beta-MCA`,
- preserve the original header exactly for traceability,
- produce multiple query candidates when the header contains a likely synonym.

This stage should never drop information from the original header. It prepares safer external queries and makes name-normalization choices auditable.

## Stage 2: Compound Identity Resolution

This stage resolves a normalized metabolite name to stable compound identities. It should use official programmatic sources and cache every query.

Recommended source order:

1. PubChem PUG-REST for first-pass name and synonym matching.
2. ChEBI REST or OLS-backed access for chemical ontology confirmation.
3. HMDB downloaded data for metabolite-specific identity and annotation enrichment.

PubChem should provide candidate `CID`, synonyms, `SMILES`, and `InChIKey`. ChEBI should provide ontology-linked chemical identity and parent terms. HMDB should enrich metabolite-specific identifiers and taxonomy where available.

The resolver should keep candidate evidence, not just a single final answer.

Recommended evidence fields:

- `source`
- `query`
- `candidate_id`
- `candidate_name`
- `synonyms`
- `smiles`
- `inchikey`
- `match_score`
- `match_reason`
- `source_url_or_dataset`
- `query_timestamp`

Recommended resolution statuses:

- `unresolved`
- `resolved_high_confidence`
- `resolved_low_confidence`
- `multi_hit_needs_review`
- `manually_curated`

High-confidence resolution should require more than a loose name hit. Preferred evidence includes exact normalized-name or synonym match plus stable structural identity, especially `InChIKey`, with no competing plausible candidates.

## Stage 3: Taxonomy And Ontology Enrichment

This stage translates resolved identity into reusable annotation columns.

Recommended fields in the enriched annotation layer:

- `pubchem_cid`
- `chebi_id`
- `hmdb_id`
- `inchikey`
- `canonical_compound_name`
- `chebi_ontology_terms`
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

The current `superclass`, `subclass`, and `pathway_tag` columns in `metabolite_annotation.csv` can remain as simplified Stage 3-facing fields, but the richer evidence should live in a cache or evidence table so lossy flattening does not destroy auditability.

## Stage 4: Rule-Based Model Membership

The rule engine should read the model registry and attempt automatic membership generation for every existing non-excluded registry entry, including current draft families.

Initial target families:

- `bile_acid`
- `indole_tryptophan`
- `phenyl_phenol`
- `tca_organic_acid`
- `nucleotide_energy`
- `broad_aromatic_union`

Future families such as `fatty_acid` and `sugars_sugar_phosphates` can be added by appending rules and registry rows.

Rules should use ontology, taxonomy, pathway tags, curated synonym lists, and structural-class tags. Name-only fallback rules are allowed only when marked as lower confidence or when they match a narrow curated synonym list.

Example rule shape:

```text
bile_acid:
  include if ChEBI term or parent term indicates bile acid
  include if HMDB or ClassyFire taxonomy indicates bile acid or bile acid derivative
  include if normalized name matches a curated bile-acid synonym list
  exclude if identity status is unresolved
  mark needs_review if multiple plausible compound identities exist
```

The first implementation should favor conservative precision over broad recall.

## Membership Output Policy

Automatic high-confidence memberships may enter `model_membership.csv` so Stage 3 can run without detailed manual review.

They must not be labeled as manually curated.

Recommended output values:

```text
membership_source = auto_taxonomy_rule
review_status = auto_high_confidence
ambiguous_flag = false
notes = include rule name and compact evidence pointer
```

Low-confidence, unresolved, or multi-hit candidates should not enter the default membership table. They should go to a review queue:

```text
review_status = needs_review
ambiguous_flag = true
```

Manual decisions can later promote rows to:

```text
review_status = manually_curated
membership_source = manual_review
```

For the current early-stage project, `auto_high_confidence` rows can be used for exploratory Stage 3 RSA and provisional model ranking. Reports must label those memberships as auto-seeded and should not treat them as fully reviewed biological claims.

If an auto-seeded model is included in a primary-tier registry row, the result should still be reported as provisional until the membership rows are manually accepted. This keeps the pipeline useful before detailed review while preventing automatic taxonomy matches from being presented as curated biochemical conclusions.

## Caching And Reproducibility

All external calls and downloaded datasets should be cached locally under a versioned cache directory, for example:

```text
data/<batch>/model_space_cache/
```

Recommended cache contents:

- normalized header table,
- PubChem response cache,
- ChEBI or OLS response cache,
- HMDB parsed lookup cache,
- final identity-resolution evidence table,
- taxonomy enrichment evidence table,
- model-membership rule evidence table,
- review queue.

The generated `model_space` directory should include a manifest with:

- input `matrix.xlsx` path and file hash,
- input trial metadata path and file hash,
- rule version,
- cache version,
- generation timestamp,
- external source names and versions where available.

This prevents later runs from silently changing when upstream databases update.

## Error Handling

The builder should fail fast when:

- `matrix.xlsx` has duplicate or empty metabolite headers after normalization,
- `stimulus_sample_map.csv` would map to sample IDs absent from `matrix.xlsx`,
- the registry has duplicate model IDs,
- a rule references an unknown model ID,
- generated membership references a metabolite absent from the matrix.

The builder should not fail the whole run when individual metabolites are unresolved. Those metabolites should be routed to the review queue.

## Stage 3 Runtime Boundary

`scripts/run_rsa.py` should remain a consumer of finalized `model_space` inputs. It should not query PubChem, ChEBI, HMDB, or ClassyFire. It should not mutate `model_space` CSVs.

A separate builder command should be responsible for generating or refreshing the model-space inputs. The builder can write draft outputs, caches, and review queues. The RSA runner should only read those outputs.

## Testing Strategy

Use deterministic unit tests for:

- header normalization,
- parenthetical alias extraction,
- candidate scoring,
- high-confidence versus multi-hit status assignment,
- rule-to-membership translation,
- review-queue routing,
- manifest creation,
- validation against `matrix.xlsx`.

External API calls should be mocked or replayed from local fixtures. Integration tests can use a tiny synthetic matrix and a cached response bundle so CI does not depend on network access.

## Open Decisions

1. Whether to store enriched annotation as extra columns in `metabolite_annotation.csv` or as a separate evidence table with a simplified CSV projection.
2. Whether to include ClassyFire through a local cache, a web API, or only through HMDB taxonomy fields for the first implementation.
3. Whether the first production builder should write into `data/20260313/model_space_202603_auto/` to avoid overwriting the historical seed directory.

## Recommended First Implementation Slice

The first implementation should create a separate auto-seeded output directory rather than mutating the current historical directory:

```text
data/20260313/model_space_202603_auto/
```

It should support:

- local header normalization,
- cached or mockable PubChem identity lookup,
- optional ChEBI and HMDB enrichment through cache files,
- rules for all existing registry draft families,
- `auto_high_confidence` membership output,
- review queue output,
- manifest output.

This slice is enough to make the model space reproducible while keeping scientific interpretation provisional.

## References

- PubChem PUG-REST documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- ChEBI: https://www.ebi.ac.uk/chebi/
- HMDB downloads: https://hmdb.ca/downloads

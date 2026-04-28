# Agent Memory

> memory_schema_version: 1
> updated_at: 2026-04-25T19:47:56+08:00
> last_consolidated: 2026-03-27

<!-- Partial migration status: `Preferences` and `Active Threads` use durable memory schema v1. Remaining sections stay in legacy form until they are migrated. -->

## Facts

<!-- Migrated from legacy Project Context bullets on 2026-04-09. -->

### MEM-20260409-010 | The project uses versioned raw and derived data roots

- type: fact
- source: state/project-status.json + prior MEMORY.md bullet
- created_at: 2026-04-09T17:59:45+08:00
- updated_at: 2026-04-09T17:59:45+08:00
- confidence: 0.9
- status: active
- tags: [project-layout, data-roots]
- topic: repository-layout
- last_seen: 2026-04-09
- aliases: [versioned datasets, data and results roots]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
This project is a C. elegans bacteria metabolite neural analysis repository with versioned raw and derived datasets organized under `data/` and `results/`.

Why it matters:
- Makes dataset lineage and output locations explicit across batches.
- Helps later sessions interpret file paths without rediscovering the repository layout.

Evidence:
- state/project-status.json
- prior MEMORY.md

### MEM-20260409-011 | The legacy batch is archived under the 202601 layout

- type: fact
- source: state/project-status.json + prior MEMORY.md bullet
- created_at: 2026-04-09T17:59:45+08:00
- updated_at: 2026-04-09T17:59:45+08:00
- confidence: 0.9
- status: active
- tags: [legacy-batch, data-layout]
- topic: repository-layout
- last_seen: 2026-04-09
- aliases: [202601 batch, legacy data layout]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The legacy batch is archived under `data/202601/data.parquet`, with derived preprocess outputs in `data/202601/processed` and reviewed Stage 1/2 outputs in `results/202601/`.

Why it matters:
- Preserves the location of the prior canonical dataset and its reviewed derived outputs.
- Prevents later work from conflating legacy outputs with the newer 202603 batch.

Evidence:
- state/project-status.json
- prior MEMORY.md

### MEM-20260409-012 | The 202603 batch uses the 20260313 raw file and versioned outputs

- type: fact
- source: state/project-status.json + prior MEMORY.md bullet
- created_at: 2026-04-09T17:59:45+08:00
- updated_at: 2026-04-09T17:59:45+08:00
- confidence: 0.9
- status: active
- tags: [202603-batch, data-layout]
- topic: repository-layout
- last_seen: 2026-04-09
- aliases: [202603 batch, 20260313 data file]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The newer batch uses `data/20260313_data.parquet`, with preprocess outputs in `data/20260313_preprocess` and versioned analysis outputs under `results/202603/`, including a Stage 1 run in `results/202603/stage1_reliability`.

Why it matters:
- Separates the current batch layout from the archived 202601 layout.
- Keeps downstream output discovery anchored to the correct batch root.

Evidence:
- state/project-status.json
- prior MEMORY.md

### MEM-20260409-013 | data/matrix.xlsx is the metabolite matrix input

- type: fact
- source: docs/whole-analysis-plan.md + prior MEMORY.md bullet
- created_at: 2026-04-09T17:59:45+08:00
- updated_at: 2026-04-09T17:59:45+08:00
- confidence: 0.7
- status: active
- tags: [matrix-input, metabolism-analysis]
- topic: model-inputs
- last_seen: 2026-04-09
- aliases: [metabolite matrix, matrix.xlsx]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
`data/matrix.xlsx` is the metabolite matrix input used for metabolism analysis, with rows representing samples and columns representing components.

Why it matters:
- Gives later stages a stable pointer to the main matrix-style model input.
- Prevents ambiguity about what `data/matrix.xlsx` represents semantically.

Evidence:
- docs/whole-analysis-plan.md
- prior MEMORY.md

### MEM-20260418-001 | data/matrix.xlsx behaves like a derived abundance-like matrix, not a closed relative-abundance table

- type: fact
- source: raw-vs-matrix investigation on 2026-04-18
- created_at: 2026-04-18T16:58:57+08:00
- updated_at: 2026-04-18T16:58:57+08:00
- confidence: 0.7
- status: active
- tags: [matrix-input, metabolism-analysis, relative-signal, compositionality]
- topic: model-inputs
- last_seen: 2026-04-18
- aliases: [matrix semantics, not total-sum normalized, derived metabolite signal]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
`data/matrix.xlsx` is strongly aligned with `data/metabolism_raw_data.xlsx` but
does not behave like a simple transpose or a per-sample total-sum-normalized
relative-abundance table. The current evidence is more consistent with a
feature-wise transformed abundance-like or relative-signal matrix.

Why it matters:
- Raw Euclidean distances on `matrix.xlsx` are vulnerable to scale and extreme
  values, but Aitchison/CLR assumptions are also not fully justified by the
  observed file semantics.
- Later RSA method choices should treat `matrix.xlsx` as abundance-like derived
  input rather than as confirmed compositional data.

Evidence:
- memory/2026-04-18.md
- data/matrix_from_raw_affine_fit.xlsx

### MEM-20260418-003 | data/metabolism_raw_data.xlsx already contains metabolite identity and taxonomy metadata for the current matrix panel

- type: fact
- source: user correction + local workbook comparison on 2026-04-18
- created_at: 2026-04-18T17:14:22.8127605+08:00
- updated_at: 2026-04-18T17:14:22.8127605+08:00
- confidence: 1.0
- status: active
- tags: [matrix-input, metabolism-analysis, raw-annotation, model-space]
- topic: model-inputs
- last_seen: 2026-04-18
- aliases: [raw metabolite metadata, preannotated metabolite identities]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
`data/metabolism_raw_data.xlsx`, sheet `all`, already includes parsed
metabolite identity and taxonomy columns such as `name`, `KEGG`, `HMDB`,
`SuperClass`, `Class`, `SubClass`, and `DirectParent`, and those rows align to
the 380 metabolite columns in `data/matrix.xlsx` after small local
canonicalization fixes.

Why it matters:
- Makes repeated external identity resolution unnecessary for the current
  dataset.
- Means future Stage 3 model-space work should join against the raw workbook
  first and reserve PubChem or ChEBI lookup for true metadata gaps only.

Evidence:
- memory/2026-04-18.md
- data/metabolism_raw_data.xlsx
- data/matrix.xlsx

### MEM-20260418-006 | The 202604 batch uses versioned preprocess and analysis outputs under data/202604 and results/202604

- type: fact
- source: 2026-04-18 preprocess/reliability/direct-RSA execution
- created_at: 2026-04-18T19:58:01.1475827+08:00
- updated_at: 2026-04-18T19:58:01.1475827+08:00
- confidence: 0.9
- status: active
- tags: [202604-batch, data-paths, workflow, rsa]
- topic: repository-layout
- last_seen: 2026-04-18
- aliases: [202604 batch, 202604 preprocess outputs, 202604 rsa outputs]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The current 202604 batch uses `data/202604/202604_data.parquet`, with
preprocess outputs at `data/202604/202604_preprocess` and refreshed analysis
outputs at `results/202604/reliability` and `results/202604/rsa`.

Why it matters:
- Keeps future reruns and reviews anchored to the correct batch-specific roots.
- Prevents later sessions from conflating the new 202604 outputs with the prior
  202603 or archived 202601 layouts.

Evidence:
- memory/2026-04-18.md
- state/project-status.json
- data/202604/202604_preprocess/qc/preprocessing_report.json
- results/202604/reliability/run_summary.json
- results/202604/rsa/run_summary.json

### MEM-20260418-007 | Reliability now writes one focus-view same-vs-different figure per date

- type: fact
- source: 2026-04-18 reliability output contract update
- created_at: 2026-04-18T21:19:45.7863800+08:00
- updated_at: 2026-04-18T21:19:45.7863800+08:00
- confidence: 0.9
- status: active
- tags: [reliability, workflow, date-generalization, data-paths]
- topic: reliability-outputs
- last_seen: 2026-04-18
- aliases: [per-date same vs different figures, date-level reliability figures]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
`reliability` now writes one focus-view same-vs-different distribution figure
per available `date` under `results/<run>/reliability/figures/`, using the
filename pattern `same_vs_different_by_date__<date>.png`.

Why it matters:
- Makes it much easier to spot weak or anomalous dates without rebuilding the
  analysis manually from pairwise comparison tables.
- Keeps the pooled main figures intact while adding a date-level diagnostic
  layer for real-data review.

Evidence:
- memory/2026-04-18.md
- src/bacteria_analysis/reliability_outputs.py
- results/202604/reliability/run_summary.json

### MEM-20260418-008 | Reliability now writes pooled and per-date per-stimulus same-vs-different figures

- type: fact
- source: 2026-04-18 reliability output contract update
- created_at: 2026-04-18T22:47:55.6330925+08:00
- updated_at: 2026-04-18T22:47:55.6330925+08:00
- confidence: 0.9
- status: active
- tags: [reliability, workflow, date-generalization, data-paths]
- topic: reliability-outputs
- last_seen: 2026-04-18
- aliases: [per-stimulus same vs different figures, pooled stimulus gap figure]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
`reliability` now writes one pooled per-stimulus same-vs-different figure and
one additional per-date per-stimulus figure under
`results/<run>/reliability/figures/`, using the filename pattern
`per_stimulus_same_vs_different__<scope>.png`.

Why it matters:
- Makes it easy to see which stimuli have weak `same < different` separation in
  the pooled data and within individual dates.
- Complements the pooled and per-date same-vs-different distribution figures
  with a stimulus-level diagnostic view.

Evidence:
- memory/2026-04-18.md
- src/bacteria_analysis/reliability_outputs.py
- results/202604/reliability/run_summary.json

### MEM-20260418-009 | A filtered 202604 rerun without 20260331 improves pooled reliability and RSA but loses LODO support

- type: fact
- source: 2026-04-18 filtered rerun excluding 20260331
- created_at: 2026-04-18T23:37:04.1645429+08:00
- updated_at: 2026-04-18T23:37:04.1645429+08:00
- confidence: 0.9
- status: active
- tags: [202604-batch, filtered-batch, reliability, rsa, data-paths]
- topic: batch-review
- last_seen: 2026-04-18
- aliases: [20260331 filtered rerun, without 20260331 batch]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
A derived 202604 comparison batch excluding `20260331` is stored at
`data/202604/202604_data_without_20260331.parquet`, with preprocess outputs at
`data/202604/202604_preprocess_without_20260331` and analysis outputs under
`results/202604_without_20260331`. In that rerun, pooled reliability and RSA
improve relative to the full 202604 batch, but `focus_view_lodo_accuracy_mean`
becomes unavailable.

Why it matters:
- Preserves a clean side-by-side comparison without overwriting the original
  202604 outputs.
- Supports treating `20260331` as a likely degraded date while keeping the loss
  of date-level generalization evidence explicit.

Evidence:
- memory/2026-04-18.md
- data/202604/202604_preprocess_without_20260331/qc/preprocessing_report.json
- results/202604_without_20260331/reliability/run_summary.json
- results/202604_without_20260331/rsa/run_summary.json

### MEM-20260419-001 | Aitchison CLR Euclidean is the strongest pooled metabolite distance method on the filtered 202604 batch

- type: fact
- source: 2026-04-19 pooled distance-method comparison on the filtered 202604 batch
- created_at: 2026-04-19T00:07:25.5658378+08:00
- updated_at: 2026-04-19T01:35:58.0019370+08:00
- confidence: 0.7
- status: superseded
- tags: [rsa, filtered-batch, matrix-input, review-workflow]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [filtered pooled distance-method comparison, Aitchison highest pooled RSA]
- ttl_days: null
- supersedes: []
- superseded_by: MEM-20260419-004

Summary:
On the filtered `202604_without_20260331` batch, the local pooled
distance-method comparison ranks `Aitchison (CLR + Euclidean)` above the
current production correlation contract for `global_profile`. Its pooled RSA is
`0.150529` in `response_window` and `0.168782` in `full_trajectory`, versus
`0.126355` and `0.133776` for the current production correlation method.

Why it matters:
- Records the earlier pooled-only review state before the later QC-filtered
  non-Aitchison recheck replaced it.
- The result is still exploratory because the upstream semantics of
  `data/matrix.xlsx` are not yet confirmed to satisfy compositional
  assumptions.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/rsa_distance_method_comparison/run_summary.md

### MEM-20260419-003 | matrix.xlsx is closest to a feature-centered ratio-like signal and raw QCRSD uses fractional scale

- type: fact
- source: 2026-04-19 raw-vs-matrix recheck on the filtered 202604 workflow
- created_at: 2026-04-19T01:35:58.0019370+08:00
- updated_at: 2026-04-19T01:35:58.0019370+08:00
- confidence: 0.9
- status: active
- tags: [matrix-input, metabolism-analysis, relative-signal, distance-method, qcrsd]
- topic: model-inputs
- last_seen: 2026-04-19
- aliases: [feature-median-centered matrix, qcrsd uses fractional scale]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Across the 299 shared `A*` samples and 380 aligned metabolites, `data/matrix.xlsx`
preserves the sample ordering of the raw workbook reasonably well but is much
closer to a feature-wise centered ratio-like transform than to raw absolute
abundance. Among simple reference candidates, dividing each raw metabolite by
its across-sample median is the closest match tested. The raw workbook `QCRSD`
column is also stored on a fractional scale, so `0.2` and `0.3` correspond to
20% and 30% RSD.

Why it matters:
- Supports treating `matrix.xlsx` as a non-log ratio-like signal centered near
  1, which makes `log2(matrix)` a better-matched transformation candidate than
  `log1p` for distance review.
- Prevents later sessions from accidentally applying `QCRSD <= 20` or
  `QCRSD <= 30` as literal numeric thresholds and retaining every feature.

Evidence:
- memory/2026-04-19.md
- data/metabolism_raw_data.xlsx
- data/matrix.xlsx

## Decisions

<!-- Migrated from legacy Architecture & Design Decisions bullets on 2026-04-09. -->

### MEM-20260409-004 | Use date + worm_key + segment_index as the canonical trial identifier

- type: decision
- source: docs/superpowers/specs/2026-03-27-data-preprocessing-design.md + prior MEMORY.md bullet
- created_at: 2026-04-09T17:55:16+08:00
- updated_at: 2026-04-09T17:55:16+08:00
- confidence: 0.9
- status: active
- tags: [trial-identifier, preprocessing]
- topic: preprocessing
- last_seen: 2026-04-09
- aliases: [trial-level identifier, trial_id]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Use `date + worm_key + segment_index` as the canonical trial-level identifier.

Why it matters:
- Keeps downstream joins and trial-level reasoning consistent across preprocessing and analysis outputs.
- Prevents later stages from inventing competing trial keys.

Evidence:
- docs/superpowers/specs/2026-03-27-data-preprocessing-design.md
- prior MEMORY.md

### MEM-20260409-005 | Preserve trial-level structure before aggregation

- type: decision
- source: docs/superpowers/specs/2026-03-27-data-preprocessing-design.md + docs/whole-analysis-plan.md + prior MEMORY.md bullet
- created_at: 2026-04-09T17:55:16+08:00
- updated_at: 2026-04-09T17:55:16+08:00
- confidence: 0.9
- status: active
- tags: [trial-level-structure, preprocessing]
- topic: preprocessing
- last_seen: 2026-04-09
- aliases: [preserve trial structure, do not average early]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Preserve trial-level structure rather than averaging early across worms or presentations.

Why it matters:
- Keeps later reliability, geometry, and RSA analyses compatible with the real data structure.
- Avoids destroying variance and grouping information too early in the pipeline.

Evidence:
- docs/superpowers/specs/2026-03-27-data-preprocessing-design.md
- docs/whole-analysis-plan.md
- prior MEMORY.md

### MEM-20260416-001 | Default the user-facing workflow to direct RSA

- type: decision
- source: user instruction + direct-RSA default-flow design/implementation session
- created_at: 2026-04-16T09:26:51+08:00
- updated_at: 2026-04-16T09:26:51+08:00
- confidence: 1.0
- status: active
- tags: [rsa, workflow, direct-mode, matrix-input, model-space]
- topic: rsa-workflow
- last_seen: 2026-04-16
- aliases: [direct RSA default flow, preprocess reliability rsa, matrix-first RSA]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Default the user-facing analysis flow to `preprocess -> reliability -> rsa`.
The `rsa` step should read `matrix.xlsx` and `preprocess_root` directly by
default, while `model_space` remains an optional curated path instead of a
required intermediate artifact.

Why it matters:
- Keeps the common RSA path aligned with the user's actual goal of testing
  `global_profile` against neural representations.
- Avoids forcing `model_space` generation when subset models are not needed.

Evidence:
- memory/2026-04-16.md
- docs/superpowers/specs/2026-04-15-direct-rsa-default-flow-design.md
- docs/superpowers/plans/2026-04-15-direct-rsa-default-flow-implementation.md

### MEM-20260418-002 | Keep RSA distance-method comparison scripts in an ignored local data folder

- type: decision
- source: user instruction on 2026-04-18 + local setup
- created_at: 2026-04-18T16:58:57+08:00
- updated_at: 2026-04-18T16:58:57+08:00
- confidence: 1.0
- status: active
- tags: [rsa, local-tools, ignored-data, review-workflow]
- topic: rsa-workflow
- last_seen: 2026-04-18
- aliases: [local rsa comparison scripts, ignored comparison folder]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Keep one-off RSA distance-method comparison code in
`data/local_rsa_distance_method_comparison/` so it can be rerun and edited
locally without adding more git history to the main implementation path.

Why it matters:
- Preserves the current comparison tooling for later review or updates.
- Keeps exploratory analysis code separate from production `src/` and `scripts/`
  paths.

Evidence:
- memory/2026-04-18.md

### MEM-20260418-004 | Prefer raw-workbook metabolite annotation over external identity caches in Stage 3 model-space generation

- type: decision
- source: implementation session on 2026-04-18
- created_at: 2026-04-18T17:36:54.3318631+08:00
- updated_at: 2026-04-18T17:36:54.3318631+08:00
- confidence: 1.0
- status: active
- tags: [model-space, raw-annotation, workflow, implementation]
- topic: model-inputs
- last_seen: 2026-04-18
- aliases: [raw workbook first, cache fallback builder]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Stage 3 `build_model_space` should prefer `data/metabolism_raw_data.xlsx` as
the primary metabolite annotation source and use identity or taxonomy caches
only as fallback for rows that are missing from the raw workbook.

Why it matters:
- Aligns the implementation with the real dataset instead of preserving an
  unnecessary cache-first architecture.
- Keeps old cache and PubChem fallback paths available without making them the
  default path for the current panel.

Evidence:
- memory/2026-04-18.md
- src/bacteria_analysis/model_space_seed.py

### MEM-20260418-005 | Keep a stable reusable model registry at data/model_space/model_registry.csv

- type: decision
- source: implementation session on 2026-04-18
- created_at: 2026-04-18T18:46:52.5644010+08:00
- updated_at: 2026-04-18T18:46:52.5644010+08:00
- confidence: 1.0
- status: active
- tags: [model-space, workflow, data-paths]
- topic: model-inputs
- last_seen: 2026-04-18
- aliases: [stable model registry, reusable registry path]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Use `data/model_space/model_registry.csv` as the stable reusable registry path
for routine Stage 3 model-space generation instead of relying on batch-specific
copies.

Why it matters:
- Gives later runs one consistent `--registry` target.
- Avoids rediscovering or recreating identical registry files across batches.

Evidence:
- data/model_space/model_registry.csv

## Preferences

<!-- Migrated from legacy User Preferences bullets on 2026-04-09. -->

### MEM-20260409-001 | Prefer concise, direct, execution-focused guidance

- type: preference
- source: prior MEMORY.md bullet
- created_at: 2026-04-09T17:47:44+08:00
- updated_at: 2026-04-09T17:47:44+08:00
- confidence: 0.9
- status: active
- tags: [communication-style, execution]
- topic: user-preferences
- last_seen: 2026-04-09
- aliases: [concise guidance, direct execution-focused guidance]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Prefer concise, direct, execution-focused guidance.

Why it matters:
- Keeps collaboration efficient.
- Favors action over long framing.

Evidence:
- prior MEMORY.md

### MEM-20260409-002 | Prefer simple, maintainable, production-friendly solutions

- type: preference
- source: prior MEMORY.md bullet
- created_at: 2026-04-09T17:47:44+08:00
- updated_at: 2026-04-09T17:47:44+08:00
- confidence: 0.9
- status: active
- tags: [engineering-style, maintainability]
- topic: user-preferences
- last_seen: 2026-04-09
- aliases: [simple maintainable solutions, production-friendly solutions]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Prefer simple, maintainable, production-friendly solutions without unnecessary abstraction.

Why it matters:
- Keeps implementations easy to read, debug, and modify.
- Avoids unnecessary abstraction and overengineering.

Evidence:
- prior MEMORY.md

### MEM-20260425-001 | Prefer concise scientific figures and module-first callable APIs

- type: preference
- source: user instruction on 2026-04-25 and project AGENTS.md config
- created_at: 2026-04-25T19:47:56+08:00
- updated_at: 2026-04-25T19:47:56+08:00
- confidence: 1.0
- status: active
- tags: [scientific-plotting, figure-style, api-design, module-first, cli]
- topic: user-preferences
- last_seen: 2026-04-25
- aliases: [concise scientific plotting, simple figure labels, module-first functions, callable Python APIs, thin CLI wrappers]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
For scientific plotting in this project, prefer simple, readable figures with
understandable nouns, brief annotations, and short titles. Avoid unnecessary
comments, redundant in-figure explanations, decorative complexity, and long
titles. For future functions, prioritize modular Python-callable APIs. Keep CLI
entry points available as thin wrappers, but do not optimize primarily for
command-line ergonomics unless explicitly requested.

Why it matters:
- Keeps research figures focused on the scientific result rather than visual or
  textual clutter.
- Makes analysis code easier for the user to call directly from notebooks,
  scripts, or follow-up modules.
- Preserves CLI access without letting CLI concerns dominate internal design.

Evidence:
- AGENTS.md
- memory/2026-04-25.md

## Lessons

<!-- Migrated from legacy Lessons Learned bullets on 2026-04-09. -->
<!-- Legacy note not migrated: the old claim about the repository having no commit history appears stale and was intentionally left out pending explicit review. -->

### MEM-20260409-006 | Incomplete neuron coverage constrains downstream analysis

- type: lesson
- source: docs/superpowers/specs/2026-03-27-data-preprocessing-design.md + prior MEMORY.md bullet
- created_at: 2026-04-09T17:55:16+08:00
- updated_at: 2026-04-09T17:55:16+08:00
- confidence: 0.9
- status: active
- tags: [neuron-coverage, analysis-constraints]
- topic: preprocessing
- last_seen: 2026-04-09
- aliases: [incomplete neuron coverage]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Trial-level neuron coverage is incomplete, which constrains feasible downstream analysis methods.

Why it matters:
- Methods must tolerate missing neurons and uneven feature support across trials.
- Assumptions of full per-trial neuron coverage will produce misleading results.

Evidence:
- docs/superpowers/specs/2026-03-27-data-preprocessing-design.md
- prior MEMORY.md

### MEM-20260409-007 | Two-date batches limit date-level generalization evidence

- type: lesson
- source: docs/stage1-reliability-explained.md + prior MEMORY.md bullet
- created_at: 2026-04-09T17:55:16+08:00
- updated_at: 2026-04-09T17:55:16+08:00
- confidence: 0.7
- status: active
- tags: [date-generalization, lodo]
- topic: reliability
- last_seen: 2026-04-09
- aliases: [limited LODO, two-date batch]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
When a batch has only two dates, date-level generalization evidence is limited and LODO-style reporting can be weak or unavailable.

Why it matters:
- Prevents over-claiming date-level generalization from underpowered batches.
- Helps interpret weak or missing LODO outputs conservatively.

Evidence:
- docs/stage1-reliability-explained.md
- prior MEMORY.md

### MEM-20260409-008 | Pooled prototype views are descriptive when dates do not share stimuli

- type: lesson
- source: docs/superpowers/specs/2026-04-01-stage3-prototype-supplement-design.md + prior MEMORY.md bullet
- created_at: 2026-04-09T17:55:16+08:00
- updated_at: 2026-04-09T17:55:16+08:00
- confidence: 0.9
- status: active
- tags: [prototype-rdm, stage3]
- topic: stage3-prototype-supplement
- last_seen: 2026-04-09
- aliases: [descriptive pooled prototype view]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
For `202603`, pooled prototype views are descriptive only when the available dates do not share stimulus identities.

Why it matters:
- Prevents descriptive pooled prototype plots from being overread as cross-date evidence.
- Preserves the boundary between supplementary interpretation and primary inference.

Evidence:
- docs/superpowers/specs/2026-04-01-stage3-prototype-supplement-design.md
- prior MEMORY.md

### MEM-20260409-009 | Prototype supplement writers need per-date neural payloads preserved

- type: lesson
- source: prior MEMORY.md bullet
- created_at: 2026-04-09T17:55:16+08:00
- updated_at: 2026-04-09T17:55:16+08:00
- confidence: 0.7
- status: active
- tags: [prototype-rdm, writer-contract]
- topic: stage3-prototype-supplement
- last_seen: 2026-04-09
- aliases: [per-date neural payloads]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Prototype supplement output generation depends on preserving per-date neural prototype payloads for the writer layer.

Why it matters:
- Supplementary output writers cannot reconstruct the needed per-date prototype artifacts if upstream results drop them.
- The writer contract depends on upstream retention of the correct intermediate payloads.

Evidence:
- prior MEMORY.md

### MEM-20260413-001 | Memory daily logs must use observed local time

- type: lesson
- source: user correction + assistant follow-up
- created_at: 2026-04-13T16:21:20+08:00
- updated_at: 2026-04-13T16:21:20+08:00
- confidence: 1.0
- status: active
- tags: [memory, timestamp, process-quality]
- topic: memory-management
- last_seen: 2026-04-13
- aliases: [no placeholder memory timestamps, observed timestamp for daily logs]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Before writing or appending a `memory/YYYY-MM-DD.md` daily-log entry, read the
actual local time and use it in the heading instead of any placeholder time.
For retrospective corrections, use explicit evidence such as commit timestamps
or the local clock value and state that evidence.

Why it matters:
- Placeholder timestamps make the daily log misleading and reduce the value of
  memory as an audit trail.
- Using observed local time makes later session reconstruction reliable.

Evidence:
- `memory/2026-04-13.md`
- user correction on 2026-04-13

### MEM-20260413-002 | Git worktrees need repo-root data paths for real-data runs

- type: lesson
- source: Stage 3 model-space auto-seed implementation session
- created_at: 2026-04-13T17:38:35+08:00
- updated_at: 2026-04-13T17:38:35+08:00
- confidence: 0.9
- status: active
- tags: [worktree, data-paths, execution]
- topic: git-worktrees
- last_seen: 2026-04-13
- aliases: [worktree data roots, repo-root data paths]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Project-local Git worktrees in this repository do not include ignored `data/`
artifacts, so real-data commands from a worktree should use repo-root absolute
paths or explicit worktree-aware path resolution.

Why it matters:
- Prevents false file-not-found failures when validating real datasets from a
  worktree.
- Keeps CLI defaults and dry-run commands aligned with how ignored data is
  stored in the main workspace.

Evidence:
- `memory/2026-04-13.md`
- Stage 3 model-space auto-seed implementation session on 2026-04-13

### MEM-20260419-002 | Low RSA should be decomposed into ceiling, distance choice, and true model mismatch

- type: lesson
- source: 2026-04-19 interpretation review on the filtered 202604 batch
- created_at: 2026-04-19T01:24:02.6443903+08:00
- updated_at: 2026-04-19T01:35:58.0019370+08:00
- confidence: 0.7
- status: active
- tags: [rsa, interpretation, model-mismatch, distance-method, neural-ceiling]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [low RSA triage, do not jump to model mismatch]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Low pooled RSA should not be treated as direct evidence that the neural and
chemical models are mismatched before checking at least three other factors:
neural ceiling or data quality limits, representation or distance choices, and
batch-specific degradation. On the filtered 202604 batch, both removing
`20260331` and changing the metabolite distance contract away from the current
control increase pooled RSA.

Why it matters:
- Later sessions should avoid over-interpreting a low RSA score as model
  failure before method sensitivity and data limits are reviewed.
- The next review step should compare distance methods across `per-date`,
  `pooled all-pairs`, and `within-date-only pooled` summaries, then estimate a
  neural ceiling.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/reliability/run_summary.json
- results/202604_without_20260331/rsa/run_summary.json
- results/202604_without_20260331/rsa_distance_method_comparison/run_summary.md

### MEM-20260419-004 | On the filtered 202604 batch, pooled RSA favors QC plus log2 Euclidean more than QC alone

- type: lesson
- source: 2026-04-19 temporary QC-filtered non-Aitchison distance-method review
- created_at: 2026-04-19T01:35:58.0019370+08:00
- updated_at: 2026-04-19T01:35:58.0019370+08:00
- confidence: 0.8
- status: active
- tags: [rsa, filtered-batch, distance-method, qcrsd, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [log2 euclidean best pooled filtered batch, qc alone does not rescue current control]
- ttl_days: null
- supersedes: [MEM-20260419-001]
- superseded_by: null

Summary:
In a temporary recheck on `results/202604_without_20260331`, applying
`QCRSD <= 0.2` and then using `log2(matrix) + Euclidean` gives the strongest
pooled non-Aitchison RSA reviewed so far: `0.180855` in `response_window` and
`0.197252` in `full_trajectory`. By contrast, applying QC alone leaves the
current `log1p + correlation` control essentially unchanged near `0.126` to
`0.129`.

Why it matters:
- Shifts the exploratory follow-up away from compositional assumptions and
  toward a ratio-aware `log2 + Euclidean` chemistry space that is more
  consistent with the current matrix semantics.
- Shows that feature QC is not enough by itself; the main pooled gain comes
  from changing the distance contract, while the remaining disagreement between
  pooled and per-date summaries still needs a `within-date-only pooled` review.

Evidence:
- memory/2026-04-19.md
- data/metabolism_raw_data.xlsx
- data/matrix.xlsx
- results/202604_without_20260331/rsa/run_summary.json

### MEM-20260419-005 | Top taxonomy-subspace RSA hits are partly date-specific on the filtered 202604 batch

- type: lesson
- source: 2026-04-19 per-date taxonomy RSA follow-up
- created_at: 2026-04-19T02:28:27.0966191+08:00
- updated_at: 2026-04-19T02:28:27.0966191+08:00
- confidence: 0.8
- status: active
- tags: [rsa, taxonomy-rsa, filtered-batch, date-generalization, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [top taxonomy categories are date-sensitive, per-date taxonomy drift]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The strongest taxonomy-defined chemistry subspaces on the filtered 202604 batch
do not all generalize evenly across dates. Several pooled hits are shared by
`20260311` and `20260313`, but some are substantially date-weighted: for
example, `Pyridines and derivatives`, `Pyridinecarboxylic acids and
derivatives`, and `Pyrimidine nucleotides` are strongest on `20260311`,
whereas `Organoheterocyclic compounds`, `Purine nucleosides`, `Benzenediols`,
and many indole-related groups peak on `20260313`; `Benzenoids` is strongly
driven by `20260414`.

Why it matters:
- Prevents pooled taxonomy RSA from being over-read as a stable cross-date
  chemical principle when some categories are clearly date-sensitive.
- Supports the next follow-up of adding `within-date-only pooled` or other
  date-aware summaries before promoting any taxonomy-derived interpretation.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/taxonomy_category_rsa_qc20_log2_euclidean/taxonomy_category_rsa_summary.csv
- results/202604_without_20260331/taxonomy_category_rsa_qc20_log2_euclidean_per_date/top_taxonomy_categories_per_date_summary.csv

### MEM-20260419-006 | Distance changes help pooled RSA, but a remaining neural-chemical mismatch ceiling is still likely

- type: lesson
- source: 2026-04-19 filtered-batch method comparison and taxonomy-union follow-up
- created_at: 2026-04-19T06:46:01.9452671+08:00
- updated_at: 2026-04-19T06:46:01.9452671+08:00
- confidence: 0.8
- status: active
- tags: [rsa, interpretation, model-mismatch, distance-method, taxonomy-rsa]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [partial mismatch ceiling, distance helps but not enough]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
On the filtered 202604 pooled review, metabolite distance choice changes the
RSA materially: the current `log1p + correlation` control reaches
`0.126071 / 0.128876`, while `QCRSD <= 0.2 + log2 + Euclidean` reaches
`0.180855 / 0.197252`, and the best reviewed taxonomy-union chemistry space
reaches `0.237033 / 0.272484`. This means the neural and chemical models are
not simply unrelated. However, even the best reviewed chemistry spaces still
stay in a low-to-moderate RSA regime, so the remaining gap is unlikely to be
explained by distance choice alone.

Why it matters:
- Prevents later sessions from collapsing the interpretation into either
  "distance does not matter" or "distance fully explains the low RSA"; the
  current pooled evidence supports neither extreme.
- Supports the next follow-up of testing partial-overlap or fused chemistry
  models, since method tuning helps but does not remove the apparent alignment
  ceiling.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/rsa_method_similarity_neural_order_qc20/run_summary.md
- results/202604_without_20260331/taxonomy_model_union_qc20_log2_euclidean/run_summary.md

### MEM-20260419-007 | Weighted subspace fusion beats the best taxonomy feature union on the filtered 202604 batch

- type: lesson
- source: 2026-04-19 partial-overlap subspace fusion review
- created_at: 2026-04-19T06:55:00+08:00
- updated_at: 2026-04-19T06:55:00+08:00
- confidence: 0.8
- status: active
- tags: [rsa, taxonomy-rsa, subspace-fusion, filtered-batch, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [partial overlap subspace fusion, weighted taxonomy fusion]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
On the filtered 202604 pooled review, a non-negative weighted fusion of
disjoint taxonomy-defined subspace RDMs exceeds the previous best feature-union
benchmark. The strongest reviewed shared-weight combination is
`Pyridines and derivatives + Organic oxygen compounds + Purine nucleosides +
Indolyl carboxylic acids and derivatives`, reaching `0.259699` in
`response_window` and `0.292577` in `full_trajectory`, above the previous best
triple feature union `0.237033 / 0.272484`. Using the same members as one
feature union drops back to `0.224755 / 0.244309`.

Why it matters:
- Supports the idea that the neural model aligns better with a weighted mixture
  of partially distinct chemistry subspaces than with a single collapsed
  chemical geometry.
- Suggests the next model-space follow-up should focus on interpretable RDM
  fusion or subspace weighting instead of only searching for better single
  distances or larger feature unions.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/partial_overlap_subspace_fusion_qc20_log2_euclidean/run_summary.md
- results/202604_without_20260331/partial_overlap_subspace_fusion_qc20_log2_euclidean/fusion_vs_union_comparison.csv

### MEM-20260419-008 | Merging non-ASE L/R neurons leaves pooled neural geometry almost unchanged and only slightly improves RSA

- type: lesson
- source: 2026-04-19 neural L/R merge review on the filtered 202604 batch
- created_at: 2026-04-19T07:08:00+08:00
- updated_at: 2026-04-19T07:08:00+08:00
- confidence: 0.8
- status: active
- tags: [rsa, neural-geometry, symmetry, filtered-batch, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [lr neuron merge, symmetric neuron merge]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The current pooled neural RDM contract first averages trials within each
stimulus, then computes pairwise correlation distance on the flattened
neuron-by-time response vectors, and RSA compares aligned upper triangles with
Spearman correlation. Under a one-off symmetry review that merged all exact
`L/R` neuron pairs except `ASEL/ASER`, the neural feature count dropped from
`22` to `13` neurons, but the pooled neural RDM stayed very similar to the
original (`0.978989` Spearman in `response_window`, `0.971743` in
`full_trajectory`). RSA changed only slightly: for `QCRSD <= 0.2 + log2 +
Euclidean`, `0.180855 -> 0.183313` in `response_window` and
`0.197252 -> 0.201482` in `full_trajectory`.

Why it matters:
- Suggests left-right redundancy is not a major hidden cause of the current low
  pooled RSA; merging symmetry-related neurons does not materially change the
  neural geometry.
- Keeps the focus on representation mismatch and chemistry-space choice rather
  than expecting a large gain from simple neural dimensionality reduction.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/neural_lr_merge_review/run_summary.md
- results/202604_without_20260331/neural_lr_merge_review/rsa_comparison.csv

### MEM-20260419-009 | Worm-split diagonal crossnobis on the L/R-merged neural space reduces pooled RSA on the filtered 202604 batch

- type: lesson
- source: 2026-04-19 neural crossnobis review on the filtered 202604 batch
- created_at: 2026-04-19T08:05:00+08:00
- updated_at: 2026-04-19T08:05:00+08:00
- confidence: 0.8
- status: active
- tags: [rsa, neural-geometry, crossnobis, filtered-batch, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [neural crossnobis review, worm-split crossnobis]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
After merging non-ASE `L/R` neuron pairs, I built an exploratory neural
crossnobis-like RDM using balanced worm-level splits within each date and a
diagonal noise normalization from within-stimulus residual variance. Across
all `900` valid global split combinations, the pooled neural crossnobis RDM
substantially reduced RSA relative to the merged correlation-based neural RDM.
For the current best non-mined chemistry distance
`QCRSD <= 0.2 + log2 + Euclidean`, RSA dropped from `0.183313` to `0.052362`
in `response_window` and from `0.201482` to `0.040956` in `full_trajectory`.

Why it matters:
- Suggests the current neural-vs-chemical alignment is not being limited by a
  lack of cross-validated discriminability in the neural RDM; replacing the
  pooled correlation geometry with crossnobis hurts rather than helps.
- Moves the next exploratory step away from more aggressive neural distance
  changes and back toward chemistry-space interpretation or other simpler
  neural summaries.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/neural_crossnobis_lr_merge_review/run_summary.md
- results/202604_without_20260331/neural_crossnobis_lr_merge_review/rsa_comparison.csv

### MEM-20260419-010 | On the L/R-merged neural space, trial median improves pooled RSA modestly while peak-absolute summaries hurt it

- type: lesson
- source: 2026-04-19 neural median vs peak-summary review on the filtered 202604 batch
- created_at: 2026-04-19T08:12:00+08:00
- updated_at: 2026-04-19T08:12:00+08:00
- confidence: 0.8
- status: active
- tags: [rsa, neural-geometry, filtered-batch, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [trial median neural prototype, peak absolute neural summary]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
With non-ASE `L/R` neurons merged, replacing the current trial-mean neural
prototype with a trial-median prototype gives a small but consistent pooled RSA
gain, while collapsing each trial to per-neuron `max(abs)` peaks hurts
alignment. Against the current best non-mined chemistry distance
`QCRSD <= 0.2 + log2 + Euclidean`, trial median changes pooled RSA from
`0.183313` to `0.195004` in `response_window` and from `0.201482` to
`0.222131` in `full_trajectory`. By contrast, the peak-absolute summary falls
to `0.116085` and `0.126490`.

Why it matters:
- Supports a simple, interpretable neural-side improvement: robust trial
  aggregation helps a little without changing the neural geometry drastically.
- Suggests aggressive temporal collapse to unsigned peak magnitude throws away
  structure that still matters for neural-vs-chemical alignment.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/neural_median_peak_review_lr_merge/run_summary.md
- results/202604_without_20260331/neural_median_peak_review_lr_merge/rsa_comparison.csv

### MEM-20260419-011 | Signed peak summaries recover much of the useful neural signal lost by absolute-peak collapse

- type: lesson
- source: 2026-04-19 signed-peak neural summary review on the filtered 202604 batch
- created_at: 2026-04-19T08:20:00+08:00
- updated_at: 2026-04-19T08:20:00+08:00
- confidence: 0.8
- status: active
- tags: [rsa, neural-geometry, filtered-batch, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [signed peak neural summary, sign-preserving peak]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
On top of the non-ASE `L/R` merged neural space, defining each trial-level
neuron feature as the value at `argmax(abs(x))` within the view window
(preserving sign) performs much better than unsigned `max(abs)` peaks. Against
the current best non-mined chemistry distance `QCRSD <= 0.2 + log2 +
Euclidean`, the signed-peak neural RDM reaches `0.219257` in
`response_window` and `0.210806` in `full_trajectory`, compared with
`0.116085` and `0.126490` for unsigned peak-absolute summaries. Signed peak is
therefore better than the mean neural prototype in `response_window`, but
still slightly below the trial-median neural prototype in `full_trajectory`.

Why it matters:
- Shows that the poor performance of unsigned peak summaries was not just due
  to temporal collapse; throwing away response sign was a major loss.
- Suggests neural dynamics still contribute, but a large share of the useful
  signal can be captured by a sign-preserving amplitude summary, especially in
  the shorter `response_window` view.

Evidence:
- memory/2026-04-19.md
- results/202604_without_20260331/neural_signed_peak_review_lr_merge/run_summary.md
- results/202604_without_20260331/neural_signed_peak_review_lr_merge/rsa_comparison.csv

### MEM-20260419-012 | The previously highlighted taxonomy triple drops under the current trial-median neural reference

- type: lesson
- source: 2026-04-19 triple-union rerun under the current neural reference
- created_at: 2026-04-19T10:20:00+08:00
- updated_at: 2026-04-19T10:20:00+08:00
- confidence: 0.87
- status: active
- tags: [rsa, taxonomy-rsa, neural-reference, filtered-batch, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [current-reference triple union, biological triple union rerun]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The previously highlighted taxonomy feature union
`Class::Indoles and derivatives + SubClass::Pyridinecarboxylic acids and
derivatives + SuperClass::Organic oxygen compounds` was not stable across
neural references. When rerun against the current non-ASE `L/R`-merged
trial-median neural RDM, the same `35`-feature union gives `0.211295` in
`response_window` and `0.244861` in `full_trajectory`, down from its older
mean-based benchmark `0.237033 / 0.272484`. Even under the matched merged
trial-mean reference, the same union only reaches `0.226383 / 0.254756`.

Why it matters:
- Prevents direct comparison between old taxonomy-union values and newer
  figures that use a different neural reference.
- Confirms that biologically interpretable chemistry subspaces must be scored
  under the same neural reference used for current reporting.

Evidence:
- results/202604_without_20260331/biological_triple_union_rdm_neural_order/run_summary.md
- results/202604_without_20260331/taxonomy_union_mean_vs_median_rerank_review/run_summary.md

### MEM-20260419-013 | Taxonomy-union optima shift under trial median; the old triple ranks only 122 of 2925 triples

- type: lesson
- source: 2026-04-19 mean-vs-median taxonomy union rerank on the filtered 202604 batch
- created_at: 2026-04-19T10:35:00+08:00
- updated_at: 2026-04-19T10:35:00+08:00
- confidence: 0.9
- status: active
- tags: [rsa, taxonomy-rsa, neural-reference, trial-median, filtered-batch]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [trial-median taxonomy rerank, taxonomy-union reference dependence]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Across the `27` unique taxonomy models used in the earlier union scan, the
best combinations change once the neural reference changes. Under the current
non-ASE `L/R`-merged trial-median neural RDM, the strongest single model stays
`Class::Pyridines and derivatives` (`0.196071 / 0.245664`), but the best pair
becomes `Class::Indoles and derivatives + Class::Purine nucleosides`
(`0.241437 / 0.254687`) and the best triple becomes
`Class::Indoles and derivatives + SuperClass::Organic oxygen compounds +
Class::Purine nucleosides` (`0.252598 / 0.270183`). The earlier highlighted
triple `Indoles + Pyridinecarboxylic acids + Organic oxygen` drops to rank
`122 / 2925` under the same trial-median reference.

Why it matters:
- Biology-driven subspace selection is neural-reference-dependent rather than
  a fixed property of the chemistry panel.
- Future taxonomy figures should be reranked under the same neural reference
  instead of reusing older mean-based shortlist results.

Evidence:
- results/202604_without_20260331/taxonomy_union_mean_vs_median_rerank_review/run_summary.md
- results/202604_without_20260331/biological_subspace_individual_rdms_neural_order/run_summary.md

### MEM-20260419-014 | The current global-profile benchmark is now anchored by trial-median neural aggregation plus QC log2 Euclidean chemistry

- type: lesson
- source: 2026-04-19 filtered-batch global-profile and distance-method summary refresh
- created_at: 2026-04-19T10:50:00+08:00
- updated_at: 2026-04-19T10:50:00+08:00
- confidence: 0.9
- status: active
- tags: [rsa, global-profile, distance-method, trial-median, filtered-batch]
- topic: rsa-workflow
- last_seen: 2026-04-19
- aliases: [current global benchmark, distance-method table]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
For broad chemistry models on the filtered 202604 batch, the best accepted
non-mined benchmark is now non-ASE `L/R` merge plus trial-median neural
correlation against `QCRSD <= 0.2 + log2(matrix) + Euclidean`, reaching
`0.195004` in `response_window` and `0.222131` in `full_trajectory`. The
matching `current chemical control` remains much lower even after switching the
neural reference to trial median, reaching only `0.133997 / 0.164824`. Under
the older mean-based neural reference used for the chemistry-only method scan,
the filtered-batch distance ranking stays
`log2 + Euclidean > current log1p + correlation > direct correlation > log2 +
correlation > raw Euclidean`, with pooled values
`0.180855 / 0.197252`, `0.126071 / 0.128876`, `0.120387 / 0.090475`,
`0.104047 / 0.121279`, and `-0.012397 / 0.003424`, respectively.

Why it matters:
- Gives later sessions one stable broad-chemistry benchmark for comparison
  against taxonomy subspaces and neural-side variants.
- Separates two effects cleanly: changing the chemistry distance helps much
  more than switching trial mean to trial median, while trial median still
  adds a modest extra gain on top of the better chemistry contract.

Evidence:
- results/202604_without_20260331/neural_chemical_rdm_neural_order_panels/run_summary.md
- results/202604_without_20260331/neural_median_peak_review_lr_merge/run_summary.md
- results/202604_without_20260331/rsa_method_similarity_neural_order_qc20/run_summary.md

### MEM-20260420-001 | Base-odor anchors support localized date-pair drift rather than dominant global date effects

- type: lesson
- source: 2026-04-20 base-odor date-effect review and user interpretation
- created_at: 2026-04-20T12:02:06+08:00
- updated_at: 2026-04-20T12:02:06+08:00
- confidence: 0.86
- status: active
- tags: [rsa, date-control, base-odor, anchor-stimuli, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-20
- aliases: [base odor date effect, localized date drift, date effects not dominant]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
Repeated base-odor anchors in `data/202604/202604_data_baseodor.parquet`
suggest that date effects may exist but are not the dominant neural structure.
The base-odor RDM is primarily organized by odor identity: odor ideal Spearman
is `0.741 / 0.754` for `t0-44 / t6-20`, while date partial correlation is only
`0.021 / 0.040`. Same-odor cross-date median distances are `0.171 / 0.128`,
well below same-date different-odor medians `1.006 / 1.044`. Diacetyl is the
most stable anchor, while 1-Octanol and NaCl can be closer to each other in
some neurons and therefore show more variable cross-date distances.

Why it matters:
- Supports the interpretation that poor or unstable bacteria RSA should not be
  attributed wholesale to a severe global date artifact.
- Localized date-pair drift remains a risk, especially for specific date
  combinations, so date-controlled and date-pair-balanced reporting is still
  required.
- Base odors are useful sanity checks but cannot fully resolve bacteria panels
  where stimulus identity and date remain confounded.

Evidence:
- memory/2026-04-20.md
- results/202604_without_20260331/date_controlled_rsa_review/base_odor_date_effect__baseodor_only/run_summary.md
- results/202604_without_20260331/date_controlled_rsa_review/base_odor_date_effect__baseodor_only/tables/base_odor_ideal_model_similarity.csv
- results/202604_without_20260331/date_controlled_rsa_review/base_odor_date_effect__baseodor_only/tables/base_odor_same_odor_date_pair_summary.csv

### MEM-20260420-002 | Global profile should be treated as a weak baseline, not the main explanatory model for neural representation

- type: lesson
- source: 2026-04-20 RSA interpretation discussion after date-controlled and base-odor reviews
- created_at: 2026-04-20T12:02:06+08:00
- updated_at: 2026-04-20T12:02:06+08:00
- confidence: 0.84
- status: active
- tags: [rsa, global-profile, model-mismatch, date-control, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-20
- aliases: [global profile weak baseline, global profile partial explanation, pair composition sensitivity]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The current interpretation is that global bacterial profile is not well suited
as the primary explanatory space for the neural representation. It captures
some shared structure but performs weakly and unstably under date-controlled
comparisons. The preferred explanation for date-composition sensitivity is that
the true global-profile-to-neural correlation is weak, so changing the sampled
stimulus pairs can move RSA substantially through pair-composition effects.
This does not mean the RSA project failed; it means the stronger hypothesis
that global profile strongly explains the neural geometry is not supported.

Why it matters:
- Prevents overclaiming that global profile explains the dominant neural
  organization.
- Reframes the project from "raise global-profile RSA" to model diagnosis:
  identify which sensory, chemical, taxonomy, metabolite-subspace, or response
  features explain the neural geometry and where model-neural mismatches arise.
- Keeps the defensible conclusion as partial model-neural structure overlap,
  not full equivalence of neural and global-profile spaces.

Evidence:
- memory/2026-04-20.md
- results/202604_without_20260331/date_controlled_rsa_review/tables/rsa_all_vs_within_date_vs_cross_date.csv
- results/202604_without_20260331/date_controlled_rsa_review/date_effect_sampling_summary.md
- results/202604_without_20260331/date_controlled_rsa_review/base_odor_date_effect__baseodor_only/run_summary.md

### MEM-20260421-001 | Current model diagnosis supports partial, date-sensitive chemical-neural overlap

- type: lesson
- source: 2026-04-21 user review discussion of model diagnosis, date controls, base odors, reliability filtering, PCA, and localization
- created_at: 2026-04-21T12:09:46+08:00
- updated_at: 2026-04-21T12:09:46+08:00
- confidence: 0.9
- status: active
- tags: [rsa, model-diagnosis, localization, date-control, neural-ceiling, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-21
- aliases: [current model diagnosis interpretation, within-date stronger cross-date mixed, partial date-sensitive overlap]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The current filtered-batch interpretation is that `QC20 + log2 + Euclidean` is
the best accepted broad chemical baseline only by data-driven RSA comparison,
not because it is mechanistically proven. Within-date RSA is stronger on
average, but cross-date is not universally bad: some cross-date date-pair
blocks are comparable to or better than weaker within-date blocks. Repeated
base odors argue against a dominant global date artifact. Reliability filtering
enriches chemical-neural overlap but remains far below the neural ceiling, and
unsupervised PCA does not robustly improve the main controlled comparisons.
Overall, chemical and neural geometry share partial, local, date-sensitive
structure rather than a globally aligned full RDM.

Why it matters:
- Future analysis should not re-litigate whether cross-date is simply bad or
  whether low RSA is just neural noise; both are too simple.
- Fixed chemical RDMs should remain baselines, while learned or supervised
  models must be validated out of sample before being interpreted.
- Anchor-local and date-pair diagnostics are explanatory context, not proof of
  a complete chemical-neural alignment.

Evidence:
- memory/2026-04-21.md
- results/202604_without_20260331/model_diagnosis/run_summary.md
- results/202604_without_20260331/model_diagnosis/tables/model_scope_summary.csv
- results/202604_without_20260331/model_diagnosis/tables/anchor_localization_summary.csv
- results/202604_without_20260331/date_controlled_rsa_review/tables/rsa_all_vs_within_date_vs_cross_date.csv
- results/202604_without_20260331/date_controlled_rsa_review/tables/rsa_by_date_pair.csv
- results/202604_without_20260331/date_controlled_rsa_review/base_odor_date_effect__baseodor_only/run_summary.md

### MEM-20260422-001 | Treat single-chemical hits as diagnostics, not formal chemical models

- type: decision
- source: user correction during supervised subspace figure redesign on 2026-04-22
- created_at: 2026-04-22T01:01:14+08:00
- updated_at: 2026-04-22T14:02:35+08:00
- confidence: 1.0
- status: active
- tags: [rsa, subspace-search, metabolite-selection, interpretation, figure-contract]
- topic: rsa-workflow
- last_seen: 2026-04-22
- aliases: [single chemical diagnostic, single marker is not chemical model, multi-chemical figure contract, final selected chemical heatmap, joint consensus boxed heatmap]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
For supervised chemical subspace search, `k=1` selected metabolite results must
be treated as single-chemical marker diagnostics rather than formal chemical
model or chemical subspace evidence. Formal figures and conclusions should
separate single-marker readouts from multi-chemical model readouts. The final
selected-chemical heatmaps should emphasize deterministic all-sample results,
using the joint-consensus, off-diagonal rank-normalized, `magma`, boxed-block
RDM pair style from `joint_consensus_rdm_review` instead of process plots or
raw row-level chart summaries.

Why it matters:
- The user's scientific target is a chemical model/subspace, not a single
  chemical marker.
- Figure outputs should directly show which chemicals were selected and how the
  selected chemical RDM resembles the neural RDM on all samples.
- Validation scopes should be reported in simple named subfolders so different
  held-out definitions are not mixed.

Evidence:
- results/202604_without_20260331/supervised_subspace_search/figures/README.md
- results/202604_without_20260331/supervised_subspace_search/final_results/README.md
- results/202604_without_20260331/supervised_subspace_search/final_results/figures/all_samples_magma_pair__response_window.png
- results/202604_without_20260331/supervised_subspace_search/final_results/figures/all_samples_magma_pair__full_trajectory.png
- scripts/redraw_supervised_subspace_figures.py
- scripts/export_supervised_subspace_final_results.py

### MEM-20260422-002 | Archive the supervised subspace search as a negative route

- type: decision
- source: user review of final selected-chemical results on 2026-04-22
- created_at: 2026-04-22T14:28:00+08:00
- updated_at: 2026-04-22T14:28:00+08:00
- confidence: 1.0
- status: active
- tags: [rsa, subspace-search, metabolite-selection, interpretation, workflow]
- topic: rsa-workflow
- last_seen: 2026-04-22
- aliases: [supervised subspace archived, negative result subspace search, do not continue supervised subspace search]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The supervised chemical subspace search should be archived rather than
continued. Its selected models collapse to only 1-2 chemicals, the final
all-sample RSA remains only moderate (`0.358686` for response_window and
`0.344581` for full_trajectory), and the many held-out scope definitions make
the route hard to explain for the scientific question.

Why it matters:
- Prevents more effort being spent polishing validation-scope charts for a route
  that no longer answers the main question.
- Keeps the outputs available as an audited negative result and method-development
  record.
- Future supervised modeling, if needed, should be redesigned from first
  principles rather than extending this implementation.

Evidence:
- results/202604_without_20260331/supervised_subspace_search/ARCHIVE_DECISION.md
- results/202604_without_20260331/supervised_subspace_search/final_results/README.md
- docs/superpowers/plans/2026-04-21-subspace-search.md

### MEM-20260423-001 | Anchor subset-RSA narrative to full-space permutation evidence

- type: decision
- source: 2026-04-23 user instruction plus memory review of permutation standards and recent handoff notes
- created_at: 2026-04-23T15:27:23+08:00
- updated_at: 2026-04-23T15:35:18+08:00
- confidence: 1.0
- status: active
- tags: [rsa, interpretation, permutation, taxonomy-rsa, subspace-search]
- topic: rsa-workflow
- last_seen: 2026-04-23
- aliases: [subset chemical narrative, full-space permutation foundation, stimulus-subset robustness is not chemical-subset null, fixed subset significance is conditional on selection, reselection stability]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
For the current reporting narrative, the full-space neural-vs-chemical RDM
permutation figure remains the foundation evidence. Taxonomy/class subsets or
weighted subset-style models may be used to support the weaker claim that some
chemical subsets raise neural-chemical RSA, but only as descriptive enrichment
on top of that foundation. The current sampled-permutation robustness figure is
a stimulus-subset check, not a chemical-subset selection null. If a chemical
subset is evaluated with its own sampled robustness check and permutation test,
that can support reliability, but only relative to the exact design: fixing the
chosen subset and resampling/permuting tests conditional robustness of that
frozen subset, whereas rerunning subset selection inside each resample or null
tests selection stability and search-corrected significance.

Why it matters:
- Keeps the main result anchored to fixed-model evidence that is already part of
  the current handoff, instead of promoting mined subset outputs as the primary
  chemical-space conclusion.
- Preserves the correct validation boundary: fixed or predeclared subset models
  can reuse fixed-model label-shuffle logic, but mined subset claims still need
  matched random feature/subspace nulls or held-out validation.
- Separates three claims that are often conflated:
  fixed-subset above-chance alignment, subset-selection stability, and
  out-of-sample generalization.
- Prevents future sessions from overstating what the existing sampled-permutation
  figure already proves about subset reliability.

Evidence:
- memory/2026-04-19.md
- memory/2026-04-22.md
- memory/2026-04-23.md

## Active Threads

<!-- Migrated from legacy Active Threads bullets on 2026-04-09. -->

### MEM-20260409-003 | Stage3 prototype supplement worktree was merged and cleaned

- type: active-thread
- source: state/project-status.json + prior MEMORY.md bullets
- created_at: 2026-04-09T17:47:44+08:00
- updated_at: 2026-04-13T16:22:16+08:00
- confidence: 1.0
- status: completed
- tags: [stage3-prototype-supplement, review-merge]
- topic: stage3-prototype-supplement
- last_seen: 2026-04-13
- aliases: [paired prototype RDM supplement, prototype supplement review]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The `codex/stage3-prototype-supplement` worktree previously contained the
completed paired prototype RDM implementation through commit `919b10d`,
including per-date neural-vs-model prototype comparisons and paired pooled
prototype figures. On 2026-04-13, the user confirmed all worktrees had already
been merged to `master`, and the worktree was removed during cleanup.

Why it matters:
- This is no longer an active workstream. Later sessions should not look for
  the removed prototype supplement worktree.

Next step:
- No worktree action remains; continue follow-up work from `master`.

Evidence:
- prior MEMORY.md
- state/project-status.json
- user confirmation on 2026-04-13
- memory/2026-04-13.md

### MEM-20260420-003 | Next RSA workstream should focus on model diagnosis beyond global profile

- type: active-thread
- source: user handoff request on 2026-04-20
- created_at: 2026-04-20T12:02:06+08:00
- updated_at: 2026-04-20T20:52:48+08:00
- confidence: 0.9
- status: superseded
- tags: [rsa, model-diagnosis, global-profile, neural-ceiling, date-control]
- topic: rsa-workflow
- last_seen: 2026-04-20
- aliases: [new RSA task, model diagnosis handoff, beyond global profile]
- ttl_days: null
- supersedes: []
- superseded_by: MEM-20260420-004

Summary:
The next thread should start from the conclusion that the current RSA result is
not a project failure, but a rejection of the strong global-profile explanation.
Global profile remains a weak baseline or partial control. The next task should
diagnose which model spaces better explain neural geometry and where the
current model-neural mismatch comes from.

Recommended next steps:
- Estimate neural RDM reliability ceilings using trial split, worm split, and
  date split where feasible.
- Use date-controlled outputs as the main reporting contract: all-pairs,
  within-date, cross-date, date-pair-balanced RSA, date-preserving permutation,
  and base-odor anchor sanity checks.
- Run residual/error analysis on stimulus pairs where neural and model RDMs
  disagree most strongly.
- Treat global profile as a baseline and compare against more sensory-proximal
  feature spaces such as volatile/odorant cues, selected metabolite subspaces,
  taxonomy-informed chemistry, response motifs, and neuron-specific tuning.
- For future experiments, include repeated bacteria anchors in addition to base
  odors so date and stimulus identity can be separated directly.

Evidence:
- MEM-20260420-001
- MEM-20260420-002
- memory/2026-04-20.md

### MEM-20260420-004 | After model diagnosis, the next RSA thread should localize shared structure rather than only report pooled overlap

- type: active-thread
- source: user-approved post-diagnosis interpretation on 2026-04-20
- created_at: 2026-04-20T20:52:48+08:00
- updated_at: 2026-04-21T12:09:46+08:00
- confidence: 0.9
- status: superseded
- tags: [rsa, model-diagnosis, shared-structure, localization, date-control]
- topic: rsa-workflow
- last_seen: 2026-04-21
- aliases: [shared structure localization, anchor-local diagnostics, local neighborhood overlap]
- ttl_days: null
- supersedes: [MEM-20260420-003]
- superseded_by: MEM-20260421-002

Summary:
The model-diagnosis package now makes the main interpretation sharper: neural
ceilings are high enough that the weak RSA cannot be explained away as noise,
and reliable-pair filtering raises the scores but still leaves a large gap to
the ceiling. This means the chemical and neural spaces share some nonrandom
structure, but the overlap is partial and should now be localized rather than
only summarized by pooled RSA or residual tables.

Recommended next steps:
- Add a new Section 3 to the model-diagnosis spec for shared-structure
  localization.
- Restrict the first pass to `global_profile_default_correlation`,
  `global_qc20_log2_euclidean`, and `best_weighted_fusion_fixed_weights`.
- Use anchor-local diagnostics on the primary reliable-pair pool (`high +
  medium`): anchor-rank alignment, top-k neighborhood overlap, triplet
  consistency, and within-date versus cross-date splits.
- Keep `high`-only analyses as a supplement rather than the main reporting
  layer.
- Prefer a compact summary-table and hotspot-figure contract over a large new
  panel of per-model figures.

Evidence:
- results/202604_without_20260331/model_diagnosis/run_summary.md
- MEM-20260420-003
- memory/2026-04-20.md

### MEM-20260421-002 | Next RSA workstream should build a supervised neural-aligned model with held-out validation

- type: active-thread
- source: user decision on 2026-04-21 after reviewing cross-date and localization interpretation
- created_at: 2026-04-21T12:09:46+08:00
- updated_at: 2026-04-21T14:33:49+08:00
- confidence: 1.0
- status: superseded
- tags: [rsa, supervised-alignment, neural-aligned-model, model-diagnosis, validation, date-control]
- topic: rsa-workflow
- last_seen: 2026-04-21
- aliases: [supervised neural-aligned model, chemical to neural mapping, neural-like RDM]
- ttl_days: null
- supersedes: [MEM-20260420-004]
- superseded_by: MEM-20260421-003

Summary:
The next most valuable RSA direction is a supervised neural-aligned model. The
preferred first pass should map chemical features to neural responses, then
build a predicted neural-like RDM from held-out predictions: `QC20 + log2`
chemical features as `X`, aggregated neural response vectors as `Y`, a simple
ridge projection implemented without heavy dependencies, and predicted
neural-like RDMs scored through the existing model-diagnosis and localization
contracts.

Recommended validation:
- Primary validation should be leave-one-date-out if feasible, because date
  generalization is the central unresolved risk.
- Secondary validation should use held-out stimulus splits to test whether the
  learned mapping generalizes beyond the training stimuli.
- Training-set predicted RDMs must not be interpreted as evidence; the main
  result should come from held-out predictions to avoid leakage and subset
  selection bias.

Next step:
- Write a concise supervised neural-aligned model design/spec before
  implementation, then add the model as another RDM source in the existing
  `model_diagnosis` pipeline rather than creating a disconnected analysis path.

Evidence:
- MEM-20260421-001
- MEM-20260420-004
- memory/2026-04-21.md

### MEM-20260421-003 | Supervised chemical subspace search branch is ready for review

- type: active-thread
- source: user request and implementation plan written on 2026-04-21
- created_at: 2026-04-21T14:33:49+08:00
- updated_at: 2026-04-23T15:27:23+08:00
- confidence: 1.0
- status: completed
- tags: [rsa, supervised-alignment, subspace-search, metabolite-selection, validation, date-control]
- topic: rsa-workflow
- last_seen: 2026-04-23
- aliases: [supervised chemical subspace search, metabolite subset search, neural-aligned candidate chemical subspace]
- ttl_days: null
- supersedes: [MEM-20260421-002]
- superseded_by: null

Summary:
The supervised chemical subspace and metabolite subset search is implemented on
branch `codex/supervised-subspace-search` at commit `f3ea2a4`, based on
`codex/model-diagnosis`. The implementation follows
`docs/superpowers/plans/2026-04-21-subspace-search.md`: taxonomy category
search is the coarse interpretable layer, within-taxonomy greedy metabolite
refinement is the main fine-grained layer, and global metabolite greedy search
is exploratory. Main evidence comes from held-out date, held-out stimulus,
cross-view validation, and matched random nulls. On 2026-04-22, the route was
closed as an audited negative result rather than advanced as the main reporting
path.

Why it matters:
- Preserves the implemented search as a reproducible method-development record.
- Reminds later sessions that this route no longer drives the main narrative,
  because final selected models were too collapsed and too hard to explain for
  the scientific question.

Next step:
- No further branch action remains; keep the archived outputs for audit and do
  not reopen this route as the primary chemical-space conclusion.

Evidence:
- docs/superpowers/plans/2026-04-21-subspace-search.md
- src/bacteria_analysis/supervised_subspace_search.py
- scripts/supervised_subspace_search.py
- tests/test_supervised_subspace_search.py
- results/202604_without_20260331/supervised_subspace_search/run_summary.md
- MEM-20260421-001
- MEM-20260421-002
- memory/2026-04-21.md

### MEM-20260423-002 | Start the next taxonomy thread from stable class-neural association rather than paper-ready reporting

- type: active-thread
- source: user instruction on 2026-04-23 after clarifying robustness and search-corrected significance
- created_at: 2026-04-23T15:45:04+08:00
- updated_at: 2026-04-23T21:35:02+08:00
- confidence: 1.0
- status: active
- tags: [rsa, taxonomy-rsa, permutation, validation, interpretation]
- topic: rsa-workflow
- last_seen: 2026-04-23
- aliases: [stable taxonomy classes, taxonomy-first stable classes, class-neural association]
- ttl_days: null
- supersedes: []
- superseded_by: null

Summary:
The next thread should start from the existing taxonomy results and ask which
chemical classes are stably associated with neural space. Keep the previously
agreed three-layer evidence split: fixed-subset above-chance alignment,
reselection stability under resampling, and search-corrected significance under
full-search permutations. Do not jump to paper-ready wording from the current
results.

For the RDM comparison figure, keep the display simple: `magma` heatmaps using
the full stimulus set in the same order and raw RDM values. Include neural,
full-chemical, and top-class chemical RDMs; do not use within-RDM scaling or add
pairwise-rank scatter panels.

The RDM comparison figure should also be mirrored into
`results/202604_without_20260331/date_controlled_rsa_review/figures` as a
companion to the date-controlled neural-chemical RDM foundation figures.

For class-to-class chemical RDM similarity, prefer a clustered class-by-class
RSA matrix rather than a histogram. Highlight `Purine nucleosides` directly and
show its similarities to other classes as a simple bar summary.

Why it matters:
- Narrows the exploratory space to interpretable taxonomy classes instead of
  reopening the archived supervised subset route.
- Makes the goal explicit: not "which class gives the highest one-off RSA",
  but "which classes recur under repeated reselection and survive a search-
  corrected null."
- Keeps future sessions aligned with the user's current threshold for evidence:
  current results are still below paper-output standard.

Next step:
- Review the generated taxonomy class stability figures and decide which
  classes should remain as stable exploratory candidates.

Evidence:
- memory/2026-04-23.md
- MEM-20260423-001
- MEM-20260419-005
- MEM-20260419-007
- docs/superpowers/plans/2026-04-23-taxonomy-class-stability-figures-plan.md
- scripts/taxonomy_class_stability_review.py
- results/202604_without_20260331/taxonomy_class_stability_review/run_summary.md
- results/202604_without_20260331/taxonomy_class_stability_review/figures/03_top_class_rdm_comparison.png
- results/202604_without_20260331/taxonomy_class_stability_review/figures/04_taxonomy_class_stability_summary.png
- results/202604_without_20260331/taxonomy_class_stability_review/figures/05_class_chemical_rdm_similarity_matrix.png
- results/202604_without_20260331/date_controlled_rsa_review/figures/neural_chemical_rdm_foundation__response_window_full_vs_top_class_rdms.png

## Key References
- `docs/neuron_data_format.md`
- `docs/ChatGPT-Guidance.md`
- `data/202601/data.parquet`
- `data/20260313_data.parquet`
- `data/20260313_preprocess/trial_level/trial_metadata.parquet`
- `data/matrix.xlsx`
- `data/model_space_202603`
- `results/202601/stage1_reliability`
- `results/202601/stage2_geometry`
- `results/202603/stage1_reliability`
- `results/202603/stage2_geometry`
- `results/202603/stage3_rsa`

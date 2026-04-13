# MEMORY Durable Schema Migration Plan

> **For agentic workers:** Do not do a big-bang rewrite. Migrate the live memory one section at a time, keep each step committable, and keep [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) usable for recall throughout the transition.

**Goal:** Gradually evolve the live [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) from the current section-and-bullet format into the new durable memory schema without losing meaning, breaking recall, or fabricating metadata.

**Architecture:** Treat the current [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) as the durable memory source of truth during migration. Migrate one section at a time into entry-based durable memory with stable IDs and explicit metadata. Keep [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) in sync only for migrated content. Leave unmigrated sections readable until their replacement is complete.

**Tech Stack:** Markdown files, JSON state files, git diff/review, manual validation

---

## Current State

- [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) uses legacy top-level sections:
  - `Project Context`
  - `Architecture & Design Decisions`
  - `User Preferences`
  - `Lessons Learned`
  - `Active Threads`
  - `Key References`
- Entries are currently plain bullets, not durable-memory records with IDs and metadata.
- [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) exists, but only as a conservative bootstrap:
  - section pointers
  - one active-thread pointer
  - no durable IDs, tag map, or aliases derived from old bullets
- [state/project-status.json](H:/Process_temporary/WJH/bacteria_analysis/state/project-status.json) and [state/heartbeat-state.json](H:/Process_temporary/WJH/bacteria_analysis/state/heartbeat-state.json) already exist and should continue to carry machine-readable current truth.

## Migration Constraints

- Do not rewrite the entire file in one pass.
- Do not invent durable IDs for every old bullet up front.
- Do not assign high confidence to old entries unless the source supports it.
- Do not let [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) get ahead of the migrated durable entries.
- Do not move current operational status into durable memory if it belongs in `state/*.json`.
- Keep the file human-readable throughout migration.
- Preserve meaning over formatting purity.

## Section Mapping

- `Architecture & Design Decisions` -> migrate to `## Decisions`
- `User Preferences` -> migrate to `## Preferences`
- `Lessons Learned` -> migrate to `## Lessons`
- `Active Threads` -> migrate to `## Active Threads`
- `Project Context` -> split carefully:
  - stable cross-session facts -> `## Facts`
  - current status or active progress -> keep in `state/*.json`
  - mixed bullets -> keep legacy until manually disambiguated
- `Key References` -> keep as a legacy appendix initially; do not force it into durable entry form in the first pass

## Confidence and Metadata Policy for Legacy Migration

When migrating an old bullet into the new schema:

- Use `confidence: 1.0` only if the old memory clearly reflects an explicit user instruction or current machine-readable truth.
- Use `confidence: 0.9` for repeated stable preferences or facts that are clearly established in the repo history or durable memory.
- Use `confidence: 0.7` for well-supported inferred lessons or decisions extracted from prior work.
- Do not migrate an old bullet into durable memory if it can only justify `confidence: 0.5` or lower. Leave it in the legacy section or daily logs until confirmed.
- When source precision is limited, cite the nearest stable source honestly, for example:
  - `source: prior MEMORY.md bullet`
  - `source: state/project-status.json + prior MEMORY.md`

## Sprint 1: Prepare a Safe Partial-Migration Layout

**Goal:** Add the minimum scaffolding needed for gradual migration without changing the meaning of the current file.

**Demo/Validation:**

- [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) remains readable in its current form.
- A reviewer can tell which sections are migrated and which are still legacy.
- No durable IDs or metadata are invented yet for content that has not been reviewed.

### Task 1.1: Add File-Level Version Markers

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md)
- **Description:** Add the file-level durable memory header required by the new schema:
  - `# Agent Memory`
  - `> memory_schema_version: 1`
  - `> updated_at: ...`
- **Dependencies:** none
- **Acceptance Criteria:**
  - The file keeps its current content and order.
  - The header reflects the new schema version without forcing immediate entry conversion.
- **Validation:**
  - Manual read-through
  - `rg "memory_schema_version" MEMORY.md`

### Task 1.2: Mark Legacy vs Migrated Zones

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md)
- **Description:** Add light structural comments or section notes so future edits can distinguish:
  - migrated durable-entry sections
  - still-legacy bullet sections
- **Dependencies:** Task 1.1
- **Acceptance Criteria:**
  - A future worker can migrate one section without rereading the entire history.
  - The file is still clean enough for normal recall.
- **Validation:**
  - Manual review of headings and comments

### Task 1.3: Freeze the Initial Migration Order

- **Location:** [docs/superpowers/plans/2026-04-09-memory-durable-schema-migration-plan.md](H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/plans/2026-04-09-memory-durable-schema-migration-plan.md)
- **Description:** Use the sequence below and do not reorder casually:
  1. `User Preferences`
  2. `Active Threads`
  3. `Architecture & Design Decisions`
  4. `Lessons Learned`
  5. `Project Context`
  6. `Key References`
- **Dependencies:** none
- **Acceptance Criteria:**
  - Migration starts with the smallest, highest-confidence sections.
- **Validation:**
  - Plan review

## Sprint 2: Migrate the Safest Sections First

**Goal:** Convert the highest-confidence sections into full durable entries while leaving the rest of the file untouched.

**Demo/Validation:**

- `User Preferences` and `Active Threads` are represented as durable entries with IDs and metadata.
- [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) contains only metadata for these migrated entries.
- Unmigrated sections still remain in legacy form and still read cleanly.

### Task 2.1: Migrate `User Preferences`

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md), [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md)
- **Description:** Convert each current user preference bullet into an entry under `## Preferences` using durable IDs and explicit metadata.
- **Dependencies:** Sprint 1
- **Acceptance Criteria:**
  - Each preference becomes one durable entry.
  - Confidence is justified conservatively.
  - `tags` and `aliases` are added only when obvious from the text.
- **Validation:**
  - IDs are unique
  - each entry has all required fields
  - [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) gets matching metadata updates

### Task 2.2: Migrate `Active Threads`

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md), [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md), optionally [state/project-status.json](H:/Process_temporary/WJH/bacteria_analysis/state/project-status.json)
- **Description:** Convert the current active-thread bullets into one or more `active-thread` durable entries. Keep the next step explicit. If part of the content is better treated as current status, keep that aspect in state.
- **Dependencies:** Task 2.1
- **Acceptance Criteria:**
  - The live active thread has a durable ID and `status: active`.
  - The `next` step in [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) matches the durable memory entry.
  - No duplicate thread is created for the same workstream.
- **Validation:**
  - Manual compare of old bullet vs new durable entry
  - index pointer and active-thread entry agree

### Task 2.3: Keep Legacy Summaries During the Transition

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md)
- **Description:** During the first migrated sections, keep a short legacy-compatible summary or pointer so recall does not abruptly depend on the new schema before the whole file catches up.
- **Dependencies:** Tasks 2.1-2.2
- **Acceptance Criteria:**
  - A reader unfamiliar with the new schema can still understand the file.
  - There is no long duplicated paragraph between legacy and migrated content.
- **Validation:**
  - Manual readability review

## Sprint 3: Migrate Decisions and Lessons

**Goal:** Convert the medium-confidence sections into durable entries without over-normalizing them.

**Demo/Validation:**

- `Architecture & Design Decisions` becomes `## Decisions`
- `Lessons Learned` becomes `## Lessons`
- Each migrated entry preserves the original claim and rationale

### Task 3.1: Convert Architecture Bullets into `decision` Entries

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md), [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md)
- **Description:** Convert each stable architecture bullet into a `decision` entry. Keep rationale concise and avoid retroactively embellishing it.
- **Dependencies:** Sprint 2
- **Acceptance Criteria:**
  - Each decision entry states the rule clearly.
  - Confidence reflects the evidence actually available.
  - If two bullets are really the same decision, merge rather than create duplicates.
- **Validation:**
  - Manual semantic diff
  - unique ID check

### Task 3.2: Convert Durable Lessons Only

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md), optionally [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md)
- **Description:** Convert only the lessons that are clearly reusable across sessions. If a lesson is really time-sensitive or status-heavy, keep it legacy or move the current aspect to state instead.
- **Dependencies:** Task 3.1
- **Acceptance Criteria:**
  - Each migrated lesson is future-useful.
  - Time-sensitive details are not accidentally frozen as timeless facts.
- **Validation:**
  - Review each entry against the write gate and confidence rubric

## Sprint 4: Resolve Context and References Carefully

**Goal:** Finish the tricky sections without forcing weak mappings.

**Demo/Validation:**

- Stable project facts live under `## Facts`
- current operational truth remains in `state/*.json`
- `Key References` is either retained as a lean appendix or deliberately split later, not force-fit into the durable entry schema

### Task 4.1: Split `Project Context` into Facts vs Current Truth

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md), [state/project-status.json](H:/Process_temporary/WJH/bacteria_analysis/state/project-status.json)
- **Description:** Review each project-context bullet and classify it:
  - stable fact -> migrate to `## Facts`
  - current status -> leave or move into `state/*.json`
  - mixed bullet -> split only if the boundary is obvious
- **Dependencies:** Sprint 3
- **Acceptance Criteria:**
  - `Facts` contains stable knowledge, not stale status text.
  - Current project phase and tasks stay machine-readable in state.
- **Validation:**
  - Cross-check [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) against [state/project-status.json](H:/Process_temporary/WJH/bacteria_analysis/state/project-status.json)

### Task 4.2: Leave `Key References` as a Controlled Legacy Appendix

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md), [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md)
- **Description:** Keep `Key References` as a short appendix or section pointer unless there is a separate approved design for reference handling. Do not force file paths into durable memory records just to eliminate the old heading.
- **Dependencies:** Task 4.1
- **Acceptance Criteria:**
  - References remain discoverable.
  - The migration does not invent a fake schema for file lists.
- **Validation:**
  - Manual review of the final section layout

## Sprint 5: Tighten the Post-Migration Rules

**Goal:** Once the main sections are migrated, lock in the new editing discipline so the file does not regress.

**Demo/Validation:**

- New durable entries use IDs and metadata
- [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) only reflects migrated durable memory
- No new legacy bullet lists are appended to migrated sections

### Task 5.1: Remove Transitional Comments That Are No Longer Needed

- **Location:** [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md)
- **Description:** After each section is fully migrated and readable, remove temporary migration comments that no longer add value.
- **Dependencies:** Sprint 4
- **Acceptance Criteria:**
  - The final file remains readable and compact.
  - Only useful scaffolding remains.
- **Validation:**
  - Manual read-through

### Task 5.2: Tighten Index Maintenance for Migrated Entries Only

- **Location:** [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md)
- **Description:** Once more sections have durable IDs, add only the clearly supported tags, aliases, and recent changes. Do not backfill speculative metadata for legacy content.
- **Dependencies:** Task 5.1
- **Acceptance Criteria:**
  - The index stays sparse and trustworthy.
  - Every index entry points to real migrated durable memory.
- **Validation:**
  - Manual pointer check
  - `rg "MEM-" memory/INDEX.md MEMORY.md`

## Validation Checklist

- [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md) stays readable before, during, and after migration.
- Every migrated entry has:
  - a stable durable ID
  - required metadata fields
  - justified confidence
  - a truthful source field
- No two durable entries describe the same unchanged rule without a merge rationale.
- `supersedes` and `superseded_by` links are symmetric when used.
- [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) is updated only for migrated entries and never outruns durable memory.
- [state/project-status.json](H:/Process_temporary/WJH/bacteria_analysis/state/project-status.json) still holds current machine-readable truth.

## Potential Risks and Gotchas

- **Over-migration:** forcing weak legacy bullets into durable entries will make the new schema look more complete than it really is.
- **Confidence inflation:** old bullets may feel authoritative because they are in `MEMORY.md`, but the true source may still be ambiguous.
- **Context/status mixing:** some `Project Context` lines are really current status snapshots and should not become timeless facts.
- **Index drift:** if [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) is backfilled too aggressively, it will stop being trustworthy.
- **Readability regression:** a half-migrated file can become harder to scan than either the old or the new format if duplication is not controlled.

## Rollback Plan

- Migrate one section per commit where possible.
- If a migration step is unsound, revert only that section and the matching [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) edits.
- Do not roll back unrelated migrated sections.
- Keep git diffs small enough that a reviewer can compare the old bullet text against the new durable entries line by line.

## Recommended First Execution Slice

The safest first real migration is:

1. add the file-level schema header to [MEMORY.md](H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md)
2. convert `User Preferences` into durable `preference` entries
3. convert `Active Threads` into one durable `active-thread` entry
4. update [memory/INDEX.md](H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md) only for those migrated entries
5. stop and review readability before touching `Decisions`, `Lessons`, or `Project Context`

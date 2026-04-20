# Shared Metabolite Cache Design (Superseded)

Date: 2026-04-15
Status: Superseded on 2026-04-18
Topic: Archived design for a project-level reusable metabolite identity and taxonomy cache

## Why This Was Superseded

This design assumed Stage 3 still needed a separate identity-resolution and
taxonomy-cache layer for the current metabolite panel. That assumption does not
hold for the actual project data.

`data/metabolism_raw_data.xlsx`, sheet `all`, already contains one row per
metabolite with parsed annotation columns, including:

- `name`
- `KEGG`
- `HMDB`
- `SuperClass`
- `Class`
- `SubClass`
- `DirectParent`

A local verification on 2026-04-18 showed:

- `data/matrix.xlsx` contains 380 metabolite columns
- `data/metabolism_raw_data.xlsx` contains 380 metabolite rows
- the two files align one-to-one after local name canonicalization
- the remaining mismatches are only punctuation or encoding variants such as
  full-width parentheses, not unresolved metabolite identities

## Decision

Do not implement a separate shared metabolite identity/taxonomy cache for the
current dataset.

For this project, the authoritative annotation source should be
`data/metabolism_raw_data.xlsx`, joined locally against `data/matrix.xlsx`.
External identity lookup should be treated as a fallback path only for future
datasets that are missing local annotation, not as the default architecture.

## Practical Follow-Up

- Reuse raw-workbook annotation directly in future Stage 3 model-space work.
- Extend local name canonicalization to handle the small set of punctuation and
  encoding mismatches between `matrix.xlsx` and the raw workbook.
- Keep this file only as an archived record of a superseded design direction.

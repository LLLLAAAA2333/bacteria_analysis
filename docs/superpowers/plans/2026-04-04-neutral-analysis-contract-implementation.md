# Neutral Analysis Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Neutralize the outward analysis contract by removing stage-prefixed public naming, migrating default output directories to neutral names, and removing main-versus-supplementary wording from RSA output artifacts while preserving current internal ranking behavior.

**Architecture:** Keep the refactor focused on the public surface: exported helpers, CLI defaults, output writers, generated run summaries, and tests. Use neutral names as the canonical API and keep compatibility aliases for existing imports so behavior remains stable while the contract shifts.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, pytest, pathlib, argparse, json

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: rename public writer helpers, switch summary title text, and preserve compatibility aliases.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`
  Responsibility: expose neutral geometry-view parsing names and neutral error text while preserving aliases.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py`
  Responsibility: rename public output helpers and neutralize summary text.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
  Responsibility: expose neutral RSA entry points and keep compatibility aliases.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: neutralize RSA writer names, figure names, and run-summary keys while translating internal tier fields into neutral outward metadata.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py`
  Responsibility: neutralize outward docstrings around prototype context.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_reliability.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
  Responsibility: switch CLI defaults and user-facing messages to neutral names.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
  Responsibility: update canonical expectations and add compatibility coverage.

## Task 1: Neutralize Public Helper Names

- [ ] Add neutral canonical function names in reliability, geometry, and RSA modules.
- [ ] Keep the existing stage-prefixed names as thin aliases to the new canonical names.
- [ ] Update docstrings and exception text to stop referring to Stage 0/1/2/3 where the message is part of the executable contract.
- [ ] Update `__all__` exports so neutral names are first-class.

## Task 2: Move Default Output Directories To Neutral Names

- [ ] Update CLI defaults and writer call sites so the canonical output directories are `reliability`, `geometry`, and `rsa`.
- [ ] Update worktree-aware path resolution in the RSA CLI to look for `results/geometry`.
- [ ] Update smoke tests and expected artifact paths to the new neutral roots.

## Task 3: Neutralize Summary Titles And RSA Figure Contract

- [ ] Change run-summary markdown titles to neutral analysis names.
- [ ] Rename RSA default figure names and run-summary keys so they no longer encode `primary` or `supplementary`.
- [ ] Keep model-ranking behavior unchanged by translating internal registry tiers into neutral outward fields.
- [ ] Rename outward prototype summary flags from `supplement_*` to neutral context wording where they are part of the output contract.

## Task 4: Update Tests For Canonical Names And Compatibility Aliases

- [ ] Update unit tests to expect the new canonical writer/helper names and neutral summary keys.
- [ ] Add assertions that compatibility aliases still resolve and behave identically where useful.
- [ ] Update CLI smoke tests so they check the neutral directory names and artifact names.

## Task 5: Verify

- [ ] Run targeted unit and CLI test suites for reliability, geometry, and RSA.
- [ ] Run the full test suite if targeted tests pass.
- [ ] Summarize any residual internal naming debt left intentionally for a future taxonomy refactor.

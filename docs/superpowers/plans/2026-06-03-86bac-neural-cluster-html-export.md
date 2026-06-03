# 86bac Neural Cluster HTML Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export an interactive Plotly HTML trajectory grid for all merged neural channels and all 86 stimuli, with stimulus columns ordered by the corrected neural RDM clustering.

**Architecture:** Add one exploratory exporter that reads the raw 86bac parquet and the current corrected `[5,25)` neural RDM with silent-neuron scale=1.0. The exporter reproduces the `visweb.py` save-HTML plotting contract: raw `delta_F_over_F0`, L/R display merge, mean +/- SEM traces, stimulus-window shading, Plotly CDN HTML, responsive configuration, no fixed figure width/height, and no zoom controls. It also writes the resolved cluster order as CSV for audit.

**Tech Stack:** Python, pandas, NumPy, SciPy hierarchical clustering, Plotly.

**Verification:** Confirm source paths, 22 raw neurons, 13 displayed neural channels, 86 RDM labels, exact RDM/data label agreement, HTML artifact read-back, cluster-order CSV read-back, Plotly trace count, stimulus-shading count, and standalone HTML structure.

---

### Task 1: Implement standalone exporter

**Files:**
- Create: `exploratory/export_86bac_neural_cluster_html.py`

- [ ] Read `data/86bac.parquet` and corrected RDM table.
- [ ] Resolve `Axxx` sample IDs from `stim_name`.
- [ ] Reproduce the non-ASE L/R merge used by `visweb.py`.
- [ ] Apply average-linkage clustering to the corrected neural RDM.
- [ ] Build a 13 x 86 Plotly subplot grid with mean +/- SEM and stimulus-window shading.
- [ ] Export CDN-backed standalone HTML and cluster-order CSV.

### Task 2: Generate and validate artifacts

**Files:**
- Create: `results/86bac_shape_pca_rsa_t05_t24_silent_scale1/figures/neural_trajectory_by_neural_rdm_cluster_order.html`
- Create: `results/86bac_shape_pca_rsa_t05_t24_silent_scale1/figures/neural_trajectory_by_neural_rdm_cluster_order.csv`
- Create: `results/86bac_shape_pca_rsa_t05_t24_silent_scale1/figures/neural_trajectory_by_neural_rdm_cluster_order.summary.json`

- [ ] Run the exporter with default inputs.
- [ ] Read back HTML, CSV, and JSON outputs.
- [ ] Confirm `13` neural rows, `86` stimulus columns, `2236` Plotly traces, and `1118` stimulus-window shapes.
- [ ] Confirm the first and last resolved stimulus labels against the cluster-order CSV.
- [ ] Record that the HTML uses raw traces for inspection while the ordering comes from the corrected active-scaled neural RDM.

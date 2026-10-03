"""Individual-based SNR and signed amplitude × template, explored at SNR ≥ 0.5.

This module reuses the reviewed individual-level fitting and cache-validation
code without modifying it. Trials are averaged within each animal first;
across-animal sample variance (ddof=1), not trial scatter, determines the gate.
Arrays retain animal × (strain, date) × cell × 40 one-second samples. The
eight-bin templates have unit RMS and coefficients retain ΔF/F₀ units.
"""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd


LEGACY_PATH = (Path(__file__).resolve().parents[2]
               / "exploration_response_profiles_20261001/code/response_representation.py")
_spec = importlib.util.spec_from_file_location("_individual_response_legacy_20261001", LEGACY_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load reviewed individual representation: {LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

CELLS = list(_legacy.CELLS)
bin_curves = _legacy.bin_curves
aggregate_strains = _legacy.aggregate_strains
_sha256 = _legacy._sha256


def fit_representation(raw, baseline_sd, strain_ids, threshold=.5, min_animals=2):
    """Fit the original individual-SNR representation at explicit new defaults.

    For n animal-mean curves, P=mean_t(mu²), V=mean_t(var_animal(ddof=1)),
    C=P−V/n, and SNR=sqrt(max(C,0)/V). The gate precedes template fitting.
    Fewer than min_animals remains missing; a measured gate failure becomes
    zero. An exactly zero measured mean has zero prediction even when it
    cannot identify a template. No trial-count threshold is imposed.
    """
    result = _legacy.fit_representation(raw, baseline_sd, strain_ids,
                                         threshold=threshold, min_animals=min_animals)
    result["eligible"] = result["counts"] >= min_animals
    return result


def load_inputs(reports_dir):
    """Run unchanged reviewed-cache provenance and original Figure 4 checks."""
    data = _legacy.load_inputs(reports_dir)
    data["metadata"] = dict(
        data["metadata"], legacy_representation_path=str(LEGACY_PATH),
        legacy_representation_sha256=_sha256(LEGACY_PATH),
        response_aggregation="Equal trials within animal, then equal animals within strain × date",
        snr_sampling_unit="Animal-mean curve; trial observations are not separate SNR samples",
    )
    return data


def _label(threshold):
    return "unfiltered" if threshold is None else f"{threshold:g}"


def _condition_table(result, conditions, cells):
    table = _legacy._condition_table(result, conditions, cells)
    table["eligible"] = result["eligible"].reshape(-1)
    return table


def _strain_audit(condition_metrics):
    """Keep date support visible after equal-date coefficient aggregation."""
    rows = []
    for (strain, cell), group in condition_metrics.groupby(["strain", "cell"], sort=True):
        rows.append(dict(
            strain=strain, cell=cell, n_dates=len(group),
            n_eligible_dates=int(group.eligible.sum()),
            n_available_dates=int(group.coefficient.notna().sum()),
            n_retained_dates=int(group.status.eq("retained").sum()),
            n_suppressed_dates=int(group.status.eq("below_snr").sum()),
            n_limited_dates=int(group.status.eq("limited_n").sum()),
            n_animals_recorded=int(group.n_animals.sum()),
            coefficient=float(group.coefficient.mean()),
            raw_coefficient=float(group.raw_coefficient.mean()),
            snr_min=float(group.snr.min()), snr_max=float(group.snr.max()),
            date_statuses=";".join(f"{row.block}:{row.status}" for row in group.itertuples(index=False)),
        ))
    return pd.DataFrame(rows)


def save_full_representation(data, out, thresholds=(None, .25, .5, .75, 1.),
                             primary=.5, min_animals=2):
    """Export the requested individual-SNR exploration and threshold controls.

    The two-animal minimum matches the immediately preceding trial exploration
    for comparable coverage; it differs from the earlier three-animal main fit.
    Thresholds are descriptive choices, not optimized against chemical data.
    """
    labels = [_label(threshold) for threshold in thresholds]
    if len(set(labels)) != len(labels) or _label(primary) not in labels:
        raise ValueError("Require unique sensitivity thresholds including the primary threshold")
    out = Path(out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    conditions, cells = data["conditions"], data["cells"]
    results, all_conditions, all_templates, summaries = {}, [], [], []
    for threshold, label in zip(thresholds, labels):
        result = fit_representation(data["raw"], data.get("baseline_sd"),
                                    [pair[0] for pair in conditions],
                                    threshold=threshold, min_animals=min_animals)
        results[label] = result
        all_conditions.append(_condition_table(result, conditions, cells).assign(threshold=label))
        all_templates.append(_legacy._template_table(result, cells).assign(threshold=label))
        for c, cell in enumerate(cells):
            status = result["status"][:, c]
            summaries.append(dict(
                threshold=label, cell=cell,
                eligible=int(result["eligible"][:, c].sum()),
                retained=int((status == "retained").sum()),
                suppressed=int((status == "below_snr").sum()),
                observed_zero=int((status == "observed_zero").sum()),
                limited=int((status == "limited_n").sum()),
                limited_n=int((status == "limited_n").sum()),
                missing=int((status == "missing").sum()),
                template_unidentified=int((status == "template_unidentified").sum()),
            ))
    main = results[_label(primary)]
    condition_metrics = _condition_table(main, conditions, cells)
    templates = _legacy._template_table(main, cells)
    values, strains = aggregate_strains(main["coefficients"], conditions)
    strain_coefficients = pd.DataFrame(values, index=pd.Index(strains, name="strain"), columns=cells)
    strain_audit = _strain_audit(condition_metrics)
    sensitivity_metrics = pd.DataFrame(summaries)
    for name, frame, index in (
            ("condition_metrics", condition_metrics, False), ("templates", templates, False),
            ("strain_coefficients", strain_coefficients, True), ("strain_audit", strain_audit, False),
            ("sensitivity_metrics", sensitivity_metrics, False),
            ("sensitivity_condition_metrics", pd.concat(all_conditions, ignore_index=True), False),
            ("sensitivity_templates", pd.concat(all_templates, ignore_index=True), False)):
        frame.to_csv(tables / f"{name}.csv", index=index)
    np.savez_compressed(tables / "representation_arrays.npz",
                        **{key: value for key, value in main.items() if isinstance(value, np.ndarray)},
                        cells=np.asarray(cells), conditions=np.asarray(conditions))
    metadata = dict(
        data["metadata"], snr_basis="individual", primary_threshold=primary,
        sensitivity_thresholds=list(thresholds), min_animals=min_animals,
        trial_count_gate=False, raw_window_seconds=[0, 40], stimulus_seconds=[0, 10],
        model_bin_seconds=5, cells=cells, response_units="delta_F_over_F0",
        gate="P=mean_t(mu^2); V=mean_t(sample variance across animal means, ddof=1); C=P-V/n; SNR=sqrt(max(C,0)/V)",
        snr_sampling_unit="Individual animal-mean 0–40 s curve; trials first averaged within animal",
        response_aggregation="Equal trials within animal, then equal animals within strain × date",
        coverage="Minimum two animals by default, retaining the latest trial-exploration sample/cell coverage; no minimum trial count",
        zero_scatter="SNR=inf when C>0; SNR=0 when C=0",
        baseline_snr="RMS condition mean / median animal baseline SD; descriptive auxiliary, not a gate",
        baseline_noise="Median trial SD over five prestimulus samples; no sqrt(n) rescaling",
        template_fit="Per cell uncentred weighted SVD of retained condition means; equal strains then equal available dates",
        template_normalization="RMS=1; greatest-absolute bin positive; first bin breaks an exact tie",
        thresholds_are="Exploratory individual-SNR settings, not calibrated detection probabilities or chemical-correlation optimization",
        coefficient_zero="Adequately observed SNR failure, or observed_zero for an exactly zero condition mean despite unidentified template; missing/limited_n remains NaN",
        raw_coefficient="Projection before zeroing onto the threshold-specific fitted template; not an independently fitted unfiltered template",
        residual_rms="Eight-bin condition mean minus gated reconstruction, in delta_F_over_F0",
        strain_aggregation="Equal available dates within strain; finite-value mean, missing remains NaN",
        provenance_verification="Unchanged legacy loader checks one-second bins/support against reviewed observations and reproduces original Figure 4 with unfiltered min_n=2",
        description_only="Full-data gate/templates are descriptive; validation refits them using training animals only",
        legacy_representation_path=str(LEGACY_PATH), legacy_representation_sha256=_sha256(LEGACY_PATH),
        code_sha256=_sha256(__file__),
    )
    (out / "representation_parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dict(primary=main, results=results, condition_metrics=condition_metrics,
                templates=templates, strain_coefficients=strain_coefficients,
                strain_audit=strain_audit, sensitivity_metrics=sensitivity_metrics, metadata=metadata)

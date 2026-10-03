"""Trial-based raw-curve SNR followed by signed amplitude × template.

Raw arrays are animal × (strain, date) × cell × 40 one-second samples.
The gate uses unbinned 0–39 s curves. Models use eight five-second means,
with no further centering, smoothing, curve normalization or cell scaling.
NaN means unavailable; an explicit zero means an adequately observed gate
failure or an exactly zero measured condition mean. Status retains the
distinction. Input measurements are never overwritten.
"""
from pathlib import Path
import hashlib
import json
import importlib.util

import numpy as np
import pandas as pd


CELLS = ["ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
         "ASEL", "ASER", "AWCON", "AWCOFF"]


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bin_curves(raw):
    """Average consecutive 5 s bins; require all 40 samples or all NaN."""
    x = np.asarray(raw, dtype=float)
    if x.ndim < 1 or x.shape[-1] != 40 or np.isinf(x).any():
        raise ValueError("Expected curves ending in 40 samples, with no infinities")
    counts = np.isfinite(x).sum(axis=-1)
    if not np.isin(counts, [0, 40]).all():
        raise ValueError("Partial raw curves require an explicit missing-data rule")
    return x.reshape(*x.shape[:-1], 8, 5).mean(axis=-1)


def _ratio(numerator, denominator):
    """Nonnegative ratio: positive / zero -> inf; zero / zero -> zero."""
    result = np.full(np.shape(numerator), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    np.divide(numerator, denominator, out=result, where=valid & (denominator > 0))
    result[valid & (denominator == 0) & (numerator > 0)] = np.inf
    result[valid & (denominator == 0) & (numerator == 0)] = 0.
    return result


def fit_representation(raw, baseline_sd, strain_ids, threshold=1.0, min_animals=2,
                       *, trial_counts, trial_second_moment, min_trials=3):
    """Fit cell templates only to conditions passing the raw-curve gate.

    Each animal has equal weight; its repeated trials share that weight.
    With m animals and r_a trials per animal, w_ar=1/(m*r_a), and
    sw2=sum_a(1/r_a)/m**2. The weighted all-trial scatter is
    V=mean_t(mean_a(E_r[y_ar(t)**2])-mu(t)**2)/(1-sw2).
    P=mean_t(mu**2), C=P-V*sw2, and SNR=sqrt(max(C,0)/V).
    C retains its sign. Scatter includes within- and between-animal variation;
    effective_n=1/sw2 describes the weights, not independent biological n.
    Baseline SNR is auxiliary. threshold=None disables the SNR gate but
    preserves minimum animal and trial requirements. Whole-animal validation
    must subset trial_counts and trial_second_moment with the same animal mask.

    Templates minimize squared error with equal strains and equal dates within
    each strain. Each template has RMS=1 and its largest-absolute bin positive.
    raw_coefficients project all available condition means onto the template
    fitted at this threshold, before gating. residual_rms describes the final
    gated reconstruction; raw_residual_rms describes the ungated projection.
    """
    x = np.asarray(raw, dtype=float)
    if x.ndim != 4:
        raise ValueError("Expected animal × condition × cell × 40 raw curves")
    bin_curves(x)
    strains = np.asarray(strain_ids, dtype=str)
    if strains.ndim != 1 or len(strains) != x.shape[1]:
        raise ValueError("Require one strain ID per condition")
    if min_animals < 1 or int(min_animals) != min_animals or min_trials < 2 or int(min_trials) != min_trials:
        raise ValueError("Require integer min_animals >= 1 and min_trials >= 2")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("SNR threshold must be finite and nonnegative, or None")
    if baseline_sd is None:
        baseline = np.full(x.shape[:-1], np.nan)
    else:
        baseline = np.asarray(baseline_sd, dtype=float)
        if baseline.shape != x.shape[:-1] or np.isinf(baseline).any():
            raise ValueError("Expected animal × condition × cell baseline SD")
        if np.any(baseline[np.isfinite(baseline)] < 0):
            raise ValueError("Baseline SD cannot be negative")

    present = np.isfinite(x).all(axis=-1)
    r = np.asarray(trial_counts, dtype=float)
    second = np.asarray(trial_second_moment, dtype=float)
    if r.shape != x.shape[:-1] or not np.isfinite(r).all() or np.any(r < 0) or np.any(r != np.floor(r)):
        raise ValueError("Trial counts must be finite nonnegative integers matching raw support")
    if not np.array_equal(r > 0, present):
        raise ValueError("Positive trial counts and complete animal curves must have identical support")
    if second.shape != x.shape or np.isinf(second).any():
        raise ValueError("Second moments must have animal × condition × cell × 40 shape")
    if not np.array_equal(np.isfinite(second), np.broadcast_to(present[..., None], x.shape)):
        raise ValueError("Second moments must be complete wherever animal means are complete")
    # Jensen's inequality also catches moment/support mistakes before fitting.
    jensen_gap = second - x**2
    jensen_tolerance = 128 * np.finfo(float).eps * np.maximum(np.abs(second), x**2)
    if np.any(jensen_gap[present] < -jensen_tolerance[present]):
        raise ValueError("Trial second moment is smaller than the squared trial mean")
    counts = present.sum(axis=0)
    n_trials = r.sum(axis=0).astype(int)
    total = np.where(present[..., None], x, 0.).sum(axis=0)
    means = np.divide(total, counts[..., None], out=np.full(x.shape[1:], np.nan),
                      where=counts[..., None] > 0)
    second_total = np.where(present[..., None], second, 0.).sum(axis=0)
    mean_second = np.divide(second_total, counts[..., None], out=np.full(x.shape[1:], np.nan),
                            where=counts[..., None] > 0)
    inverse_trials = np.divide(1., r, out=np.zeros_like(r), where=r > 0)
    sw2 = np.divide(inverse_trials.sum(axis=0), counts**2,
                    out=np.full(counts.shape, np.nan), where=counts > 0)
    effective_n = np.divide(1., sw2, out=np.full(counts.shape, np.nan), where=sw2 > 0)
    scatter_numerator = mean_second - means**2
    tolerance = 128 * np.finfo(float).eps * np.maximum(np.abs(mean_second), means**2)
    if np.any(scatter_numerator < -tolerance):
        raise ValueError("Materially negative weighted trial variance")
    # Only roundoff-sized negative variances are clipped; signed C is not clipped.
    scatter_numerator = np.maximum(scatter_numerator, 0.)
    variance = np.divide(scatter_numerator, (1. - sw2)[..., None],
                         out=np.full(x.shape[1:], np.nan), where=(1. - sw2)[..., None] > 0)
    scatter_power = variance.mean(axis=-1)
    signal_power = np.mean(means ** 2, axis=-1)
    coherent_power = signal_power - scatter_power * sw2
    snr = np.sqrt(_ratio(np.maximum(coherent_power, 0.), scatter_power))
    baseline_reference = np.full(counts.shape, np.nan)
    for k, c in np.ndindex(counts.shape):
        values = baseline[:, k, c][present[:, k, c] & np.isfinite(baseline[:, k, c])]
        if len(values):
            baseline_reference[k, c] = np.median(values)
    baseline_snr = _ratio(np.sqrt(signal_power), baseline_reference)
    mean_bins = bin_curves(means)

    eligible = (counts >= min_animals) & (n_trials >= min_trials)
    passed = eligible.copy() if threshold is None else eligible & (snr >= threshold)
    status = np.full(counts.shape, "missing", dtype="U24")
    status[(counts > 0) & (counts < min_animals)] = "limited_n"
    status[(counts >= min_animals) & (n_trials < min_trials)] = "limited_trials"
    status[eligible & ~passed] = "below_snr"
    status[passed] = "retained"
    k_count, c_count = counts.shape
    templates = np.full((c_count, 8), np.nan)
    raw_coefficients = np.full(counts.shape, np.nan)
    coefficients = np.full(counts.shape, np.nan)
    reconstruction = np.full_like(mean_bins, np.nan)
    raw_residual_rms = np.full(counts.shape, np.nan)
    template_identified = np.zeros(c_count, dtype=bool)
    template_n_conditions = passed.sum(axis=0)

    for c in range(c_count):
        rows = np.flatnonzero(passed[:, c])
        if len(rows):
            selected = mean_bins[rows, c]
            unique, inverse, block_counts = np.unique(strains[rows], return_inverse=True,
                                                       return_counts=True)
            weights = 1. / (len(unique) * block_counts[inverse])
            if np.any(selected != 0):
                _, singular_values, vt = np.linalg.svd(np.sqrt(weights[:, None]) * selected,
                                                       full_matrices=False)
                if singular_values[0] > 0:
                    h = vt[0] / np.sqrt(np.mean(vt[0] ** 2))
                    if h[np.argmax(np.abs(h))] < 0:
                        h = -h
                    templates[c] = h
                    template_identified[c] = True
                    observed = counts[:, c] > 0
                    raw_coefficients[observed, c] = mean_bins[observed, c] @ h / np.dot(h, h)
                    raw_prediction = raw_coefficients[observed, c, None] * h
                    raw_residual_rms[observed, c] = np.sqrt(np.mean(
                        (mean_bins[observed, c] - raw_prediction) ** 2, axis=-1))
        if template_identified[c]:
            coefficients[passed[:, c], c] = raw_coefficients[passed[:, c], c]
            reconstruction[passed[:, c], c] = coefficients[passed[:, c], c, None] * templates[c]
        else:
            status[passed[:, c], c] = "template_unidentified"
            # A measured condition mean that is identically zero has a known
            # zero reconstruction even though it cannot identify any shape.
            observed_zero = passed[:, c] & np.all(means[:, c] == 0., axis=-1)
            status[observed_zero, c] = "observed_zero"
            raw_coefficients[observed_zero, c] = 0.
            coefficients[observed_zero, c] = 0.
            reconstruction[observed_zero, c] = 0.
            raw_residual_rms[observed_zero, c] = 0.
        failed = eligible[:, c] & ~passed[:, c]
        coefficients[failed, c] = 0.
        reconstruction[failed, c] = 0.

    residual_rms = np.sqrt(np.mean((mean_bins - reconstruction) ** 2, axis=-1))
    return dict(means=means, mean_bins=mean_bins, counts=counts, snr=snr,
                n_trials=n_trials, effective_n=effective_n,
                sum_squared_trial_weights=sw2, eligible=eligible,
                coherent_power=coherent_power, scatter_power=scatter_power,
                signal_power=signal_power, baseline_snr=baseline_snr,
                baseline_reference=baseline_reference, status=status,
                templates=templates, template_identified=template_identified,
                template_n_conditions=template_n_conditions,
                raw_coefficients=raw_coefficients, coefficients=coefficients,
                reconstruction=reconstruction, residual_rms=residual_rms,
                raw_residual_rms=raw_residual_rms, threshold=threshold,
                min_animals=int(min_animals), min_trials=int(min_trials))


def aggregate_strains(values, conditions):
    """Return (equal-date finite means, sorted strain IDs); missing stays NaN."""
    values = np.asarray(values, dtype=float)
    if len(conditions) != len(values) or len(set(map(tuple, conditions))) != len(conditions):
        raise ValueError("Require one unique (strain, date) condition per row")
    strain_ids = np.asarray([str(pair[0]) for pair in conditions])
    strains = sorted(set(strain_ids.tolist()))
    result = np.full((len(strains), *values.shape[1:]), np.nan)
    for i, strain in enumerate(strains):
        subset = values[strain_ids == strain]
        counts = np.isfinite(subset).sum(axis=0)
        sums = np.where(np.isfinite(subset), subset, 0.).sum(axis=0)
        result[i] = np.divide(sums, counts, out=np.full(values.shape[1:], np.nan), where=counts > 0)
    return result, strains


def load_inputs(reports_dir):
    """Validate the previous caches, then add sufficient trial moments.

    The previous loader performs the reviewed-cache provenance, one-second/bin
    consistency and original Figure 4 coefficient checks. This loader reads
    cached class-trial curves, checks their means against those animal curves,
    and adds the trial counts and second moments required for weighted scatter.
    It never reads or writes the raw source or changes the older exploration.
    """
    reports = Path(reports_dir).resolve()
    legacy_path = reports / "exploration_response_profiles_20261001/code/response_representation.py"
    spec = importlib.util.spec_from_file_location("_reviewed_response_representation_20261001", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reviewed cache validation: {legacy_path}")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    data = legacy.load_inputs(reports)
    path = reports / "exploration_20260929/tables/trial_curves.parquet"
    columns = [str(t) for t in range(40)]
    trial = pd.read_parquet(path, columns=columns)
    keys = ["sample_id", "date", "worm_key", "neuron_class"]
    index = trial.index.to_frame(index=False)
    for key in keys:
        index[key] = index[key].astype(str)
    trial.index = pd.MultiIndex.from_frame(index)
    if trial.index.duplicated().any() or not np.isfinite(trial.to_numpy()).all():
        raise ValueError("Class-trial cache must have unique identities and complete finite curves")
    means = trial.groupby(level=keys).mean().sort_index()
    second = trial.pow(2).groupby(level=keys).mean().reindex(means.index)
    counts = trial.groupby(level=keys).size().reindex(means.index)
    animal_index = {name: i for i, name in enumerate(data["animals"])}
    condition_index = {pair: i for i, pair in enumerate(data["conditions"])}
    cell_index = {cell: i for i, cell in enumerate(data["cells"])}
    identities = means.index.to_frame(index=False)
    ii = [animal_index[date + "|" + worm] for date, worm in zip(identities.date, identities.worm_key)]
    kk = [condition_index[pair] for pair in zip(identities.sample_id, identities.date)]
    cc = [cell_index[cell] for cell in identities.neuron_class]
    trial_means = np.full_like(data["raw"], np.nan)
    trial_second = np.full_like(data["raw"], np.nan)
    trial_counts = np.zeros(data["raw"].shape[:-1], dtype=int)
    trial_means[ii, kk, cc] = means.to_numpy()
    trial_second[ii, kk, cc] = second.to_numpy()
    trial_counts[ii, kk, cc] = counts.to_numpy()
    if (not np.array_equal(np.isfinite(trial_means), np.isfinite(data["raw"]))
            or not np.allclose(trial_means, data["raw"], rtol=1e-12, atol=1e-14, equal_nan=True)):
        raise ValueError("Cached trial means/support do not reproduce reviewed animal curves")
    data["trial_counts"] = trial_counts
    data["trial_second_moment"] = trial_second
    data["metadata"].update(
        trial_mean_max_abs_error=float(np.nanmax(np.abs(trial_means - data["raw"]))),
        cached_class_trial_curves=len(trial),
        operational_trials=len(index[["date", "worm_key", "segment_index"]].drop_duplicates()),
        trial_count_min=int(counts.min()), trial_count_max=int(counts.max()),
        trial_cache_sha256=_sha256(path),
        legacy_validation_code_sha256=_sha256(legacy_path),
        trial_pooling="Available bilateral sides were averaged within trial/time; trials remain nested in animal")
    return data


def _label(threshold):
    return "unfiltered" if threshold is None else f"{threshold:g}"


def _condition_table(result, conditions, cells):
    rows = []
    for k, (strain, block) in enumerate(conditions):
        for c, cell in enumerate(cells):
            rows.append(dict(strain=strain, block=block, cell=cell,
                             n_animals=int(result["counts"][k, c]),
                             n_trials=int(result["n_trials"][k, c]),
                             effective_n=result["effective_n"][k, c],
                             weight_sum_squared=result["sum_squared_trial_weights"][k, c],
                             eligible=bool(result["eligible"][k, c]),
                             mean_rms=np.sqrt(result["signal_power"][k, c]),
                             scatter_rms=np.sqrt(result["scatter_power"][k, c]),
                             snr=result["snr"][k, c], baseline_snr=result["baseline_snr"][k, c],
                             status=result["status"][k, c], raw_coefficient=result["raw_coefficients"][k, c],
                             coefficient=result["coefficients"][k, c], residual_rms=result["residual_rms"][k, c],
                             coherent_power=result["coherent_power"][k, c],
                             signal_power=result["signal_power"][k, c], scatter_power=result["scatter_power"][k, c],
                             raw_residual_rms=result["raw_residual_rms"][k, c]))
    return pd.DataFrame(rows)


def _template_table(result, cells):
    return pd.DataFrame([dict(cell=cell, bin_index=t, time_s=5*t + 2.5,
                              template=result["templates"][c, t],
                              identified=bool(result["template_identified"][c]),
                              n_conditions=int(result["template_n_conditions"][c]))
                         for c, cell in enumerate(cells) for t in range(8)])


def _strain_audit(condition_metrics):
    """Record each date's contribution; do not treat a missing date as zero."""
    rows = []
    for (strain, cell), group in condition_metrics.groupby(["strain", "cell"], sort=True):
        rows.append(dict(
            strain=strain, cell=cell, n_dates=len(group),
            n_eligible_dates=int(group.eligible.sum()),
            n_available_dates=int(group.coefficient.notna().sum()),
            n_retained_dates=int(group.status.eq("retained").sum()),
            n_suppressed_dates=int(group.status.eq("below_snr").sum()),
            n_limited_dates=int(group.status.isin(["limited_n", "limited_trials"]).sum()),
            n_animals_recorded=int(group.n_animals.sum()),
            n_trials_recorded=int(group.n_trials.sum()),
            coefficient=float(group.coefficient.mean()),
            raw_coefficient=float(group.raw_coefficient.mean()),
            snr_min=float(group.snr.min()), snr_max=float(group.snr.max()),
            date_statuses=";".join(f"{row.block}:{row.status}" for row in group.itertuples(index=False))))
    return pd.DataFrame(rows)


def save_full_representation(data, out, thresholds=(None, .5, 1., 1.5, 2.), primary=1.,
                             *, min_animals=2, min_trials=3):
    """Fit/export the authorized descriptive exploration; return tables and fits."""
    out = Path(out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    conditions, cells = data["conditions"], data["cells"]
    results, all_conditions, all_templates, summaries = {}, [], [], []
    if len({_label(t) for t in thresholds}) != len(thresholds) or _label(primary) not in {_label(t) for t in thresholds}:
        raise ValueError("Require unique sensitivity thresholds including primary")
    for threshold in thresholds:
        label = _label(threshold)
        result = fit_representation(data["raw"], data["baseline_sd"], [p[0] for p in conditions],
                                    threshold, min_animals,
                                    trial_counts=data["trial_counts"],
                                    trial_second_moment=data["trial_second_moment"], min_trials=min_trials)
        results[label] = result
        all_conditions.append(_condition_table(result, conditions, cells).assign(threshold=label))
        all_templates.append(_template_table(result, cells).assign(threshold=label))
        for c, cell in enumerate(cells):
            status = result["status"][:, c]
            summaries.append(dict(threshold=label, cell=cell,
                                  eligible=int(result["eligible"][:, c].sum()),
                                  retained=int((status == "retained").sum()),
                                  suppressed=int((status == "below_snr").sum()),
                                  observed_zero=int((status == "observed_zero").sum()),
                                  limited=int(np.isin(status, ["limited_n", "limited_trials"]).sum()),
                                  limited_n=int((status == "limited_n").sum()),
                                  limited_trials=int((status == "limited_trials").sum()),
                                  missing=int((status == "missing").sum()),
                                  template_unidentified=int((status == "template_unidentified").sum())))
    main = results[_label(primary)]
    condition_metrics = _condition_table(main, conditions, cells)
    templates = _template_table(main, cells)
    values, strains = aggregate_strains(main["coefficients"], conditions)
    strain_coefficients = pd.DataFrame(values, index=pd.Index(strains, name="strain"), columns=cells)
    strain_audit = _strain_audit(condition_metrics)
    sensitivity_metrics = pd.DataFrame(summaries)
    condition_metrics.to_csv(tables / "condition_metrics.csv", index=False)
    templates.to_csv(tables / "templates.csv", index=False)
    strain_coefficients.to_csv(tables / "strain_coefficients.csv")
    strain_audit.to_csv(tables / "strain_audit.csv", index=False)
    sensitivity_metrics.to_csv(tables / "sensitivity_metrics.csv", index=False)
    pd.concat(all_conditions, ignore_index=True).to_csv(tables / "sensitivity_condition_metrics.csv", index=False)
    pd.concat(all_templates, ignore_index=True).to_csv(tables / "sensitivity_templates.csv", index=False)
    np.savez_compressed(tables / "representation_arrays.npz", **{key: value for key, value in main.items()
                        if isinstance(value, np.ndarray)}, cells=np.asarray(cells), conditions=np.asarray(conditions))
    metadata = dict(data["metadata"], primary_threshold=primary,
                    sensitivity_thresholds=list(thresholds), min_animals=min_animals, min_trials=min_trials,
                    raw_window_seconds=[0, 40], stimulus_seconds=[0, 10],
                    model_bin_seconds=5, cells=cells, response_units="delta_F_over_F0",
                    gate="Animal-equal trial weights w_ar=1/(m*r_a); sw2=sum_a(1/r_a)/m^2; P=mean_t(mu^2); V=mean_t(mean_a(E_r[y^2])-mu^2)/(1-sw2); C=P-V*sw2; SNR=sqrt(max(C,0)/V)",
                    trial_variance="Balanced across-all-trial empirical scatter; includes within- and between-animal variation",
                    effective_n="1/sum(w_trial^2); descriptive weight-effective trial count, not independent biological sample size",
                    animal_means="Unchanged equal-trial mean within animal; equal-animal mean within condition",
                    zero_scatter="SNR=inf when C>0; SNR=0 when C=0",
                    baseline_snr="RMS condition mean / median animal baseline SD; descriptive auxiliary, not a gate",
                    baseline_noise="Median trial SD over five prestimulus samples; no sqrt(n) rescaling",
                    template_fit="Per cell uncentred weighted SVD of retained condition means; equal strains then equal available dates",
                    template_normalization="RMS=1; greatest-absolute bin positive; first bin breaks an exact tie",
                    thresholds_are="Exploratory settings, not calibrated detection probabilities",
                    coefficient_zero="Adequately observed raw-curve SNR failure, or observed_zero (exactly zero condition mean despite unidentified template); missing/limited_n/limited_trials remains NaN",
                    raw_coefficient="Projection before zeroing onto the threshold-specific fitted template; not an independently fitted unfiltered template",
                    residual_rms="Eight-bin condition mean minus gated reconstruction, in delta_F_over_F0",
                    strain_aggregation="Equal available dates within strain; finite-value mean, missing remains NaN",
                    provenance_verification="One-second bins and support checked against reviewed observations; unfiltered min_n=2 fit checked against old Figure 4",
                    description_only="Full-data gate/templates are descriptive; validation must hold out whole animals and subset their trial moments before refitting",
                    code_sha256=_sha256(__file__))
    (out / "representation_parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dict(primary=main, results=results, condition_metrics=condition_metrics,
                templates=templates, strain_coefficients=strain_coefficients,
                strain_audit=strain_audit, sensitivity_metrics=sensitivity_metrics, metadata=metadata)

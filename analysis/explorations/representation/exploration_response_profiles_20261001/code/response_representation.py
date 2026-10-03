"""Exploratory raw-curve SNR gate followed by signed amplitude × template.

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
import sys

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


def fit_representation(raw, baseline_sd, strain_ids, threshold=1.0, min_animals=3):
    """Fit cell templates only to conditions passing the raw-curve gate.

    For n complete animal curves: P=mean_t(mu**2), V=mean_t(sample variance),
    C=P-V/n, SNR=sqrt(max(C,0)/V). C is retained with its sign. This is an
    exploratory across-animal signal-to-scatter ratio, not a significance test.
    Baseline SNR is auxiliary and never used by the gate. threshold=None
    disables the SNR gate while retaining the minimum animal requirement.

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
    if len(strains) != x.shape[1] or min_animals < 2 or int(min_animals) != min_animals:
        raise ValueError("Require one strain ID per condition and min_animals >= 2")
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
    counts = present.sum(axis=0)
    total = np.where(present[..., None], x, 0.).sum(axis=0)
    means = np.divide(total, counts[..., None], out=np.full(x.shape[1:], np.nan),
                      where=counts[..., None] > 0)
    residual = np.where(present[..., None], x - means[None], 0.)
    variance = np.divide(np.sum(residual ** 2, axis=0), (counts - 1)[..., None],
                         out=np.full(x.shape[1:], np.nan), where=counts[..., None] >= 2)
    scatter_power = variance.mean(axis=-1)
    signal_power = np.mean(means ** 2, axis=-1)
    noise_of_mean = np.divide(scatter_power, counts, out=np.full(counts.shape, np.nan),
                              where=counts > 0)
    coherent_power = signal_power - noise_of_mean
    snr = np.sqrt(_ratio(np.maximum(coherent_power, 0.), scatter_power))
    baseline_reference = np.full(counts.shape, np.nan)
    for k, c in np.ndindex(counts.shape):
        values = baseline[:, k, c][present[:, k, c] & np.isfinite(baseline[:, k, c])]
        if len(values):
            baseline_reference[k, c] = np.median(values)
    baseline_snr = _ratio(np.sqrt(signal_power), baseline_reference)
    mean_bins = bin_curves(means)

    eligible = counts >= min_animals
    passed = eligible.copy() if threshold is None else eligible & (snr >= threshold)
    status = np.full(counts.shape, "missing", dtype="U24")
    status[(counts > 0) & ~eligible] = "limited_n"
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
                coherent_power=coherent_power, scatter_power=scatter_power,
                signal_power=signal_power, baseline_snr=baseline_snr,
                baseline_reference=baseline_reference, status=status,
                templates=templates, template_identified=template_identified,
                template_n_conditions=template_n_conditions,
                raw_coefficients=raw_coefficients, coefficients=coefficients,
                reconstruction=reconstruction, residual_rms=residual_rms,
                raw_residual_rms=raw_residual_rms, threshold=threshold,
                min_animals=int(min_animals))


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
    """Load reviewed caches, validate provenance/support, and verify the old fit.

    Reads no raw fluorescence table. The unfiltered min_n=2 refit below is a
    compatibility check against the original Figure 4, not the new control
    (which uses min_n=3). No files are written.
    """
    reports = Path(reports_dir).resolve()
    source = reports / "response_structure_20260930"
    old = reports / "exploration_20260929"
    notebook_dir = reports.parent / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    from response_structure_display import validate_evidence

    fig4_path = reports / "response_process_draft_20261001/plot_parameters.json"
    fig4 = json.loads(fig4_path.read_text())
    if fig4["window_seconds"] != [0, 40] or fig4["bin_seconds"] != 5:
        raise ValueError("Figure 4 time contract has changed")
    input_paths = [fig4_path]
    for name, expected in fig4["source_sha256"].items():
        path = source / name
        if _sha256(path) != expected:
            raise ValueError(f"Source no longer matches Figure 4: {name}")
        input_paths.append(path)
    reviewed = validate_evidence(source)
    prep_path = source / "data/preparation_metadata.json"
    manifest_path = old / "logs/input_manifest.json"
    prep = json.loads(prep_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    old_raw_hash = next(item["sha256"] for item in manifest if item["path"] == "data/106bac.parquet")
    if old_raw_hash != prep["raw_sha256"] or prep["classes"] != CELLS:
        raise ValueError("One-second and reviewed binned caches have incompatible sources")
    curve_path = old / "tables/animal_curves.parquet"
    baseline_path = old / "tables/animal_metrics.csv"
    trial_path = old / "tables/trial_curves.parquet"
    input_paths += [prep_path, manifest_path, curve_path, baseline_path, trial_path]
    keys = ["sample_id", "date", "worm_key", "neuron_class"]
    curves = pd.read_parquet(curve_path).reset_index()
    metrics = pd.read_csv(baseline_path, dtype={key: str for key in keys})
    for key in keys:
        curves[key] = curves[key].astype(str)
    if curves.duplicated(keys).any() or metrics.duplicated(keys).any():
        raise ValueError("Duplicate animal-condition-cell cache identities")
    indexed = curves.set_index(keys).sort_index()
    metrics = metrics.set_index(keys).sort_index()
    if not indexed.index.equals(metrics.index):
        raise ValueError("Baseline metrics and animal curves have different support")
    if set(curves.neuron_class) != set(CELLS):
        raise ValueError("Expected all 13 cell classes")
    # Baseline SD is independently reconstructed from cached trial prestimulus data.
    trial_baseline = pd.read_parquet(trial_path, columns=[str(t) for t in range(-5, 0)])
    trial_baseline.index = pd.MultiIndex.from_frame(
        trial_baseline.index.to_frame(index=False).astype({key: str for key in keys}))
    baseline_check = trial_baseline.std(axis=1, ddof=1).groupby(level=keys).median().sort_index()
    if (not baseline_check.index.equals(metrics.index)
            or not np.allclose(baseline_check, metrics.baseline_sd, rtol=1e-12, atol=1e-14)):
        raise ValueError("Cached baseline SD differs from median trial prestimulus SD")

    animals = sorted((curves.date + "|" + curves.worm_key).unique())
    conditions = sorted(set(zip(curves.sample_id, curves.date)))
    animal_index = {name: i for i, name in enumerate(animals)}
    condition_index = {pair: i for i, pair in enumerate(conditions)}
    cell_index = {cell: i for i, cell in enumerate(CELLS)}
    raw = np.full((len(animals), len(conditions), len(CELLS), 40), np.nan)
    baseline = np.full(raw.shape[:-1], np.nan)
    identities = indexed.index.to_frame(index=False)
    ii = [animal_index[date + "|" + worm] for date, worm in zip(identities.date, identities.worm_key)]
    kk = [condition_index[pair] for pair in zip(identities.sample_id, identities.date)]
    cc = [cell_index[cell] for cell in identities.neuron_class]
    raw[ii, kk, cc] = indexed[[str(t) for t in range(40)]].to_numpy(float)
    baseline[ii, kk, cc] = metrics.baseline_sd.to_numpy(float)
    binned = bin_curves(raw)
    obs_path = source / "data/observations.parquet"
    obs = pd.read_parquet(obs_path)
    obs["date"], obs["block"] = obs.date.astype(str), obs.block.astype(str)
    if not obs.date.eq(obs.block).all():
        raise ValueError("Reviewed acquisition block differs from date")
    if obs.duplicated(["animal_id", "sample_id", "block", "neuron_class", "bin_index"]).any():
        raise ValueError("Duplicate reviewed observation features")
    expected = np.full_like(binned, np.nan)
    for row in obs.itertuples(index=False):
        i = animal_index[str(row.animal_id)]
        k = condition_index[(str(row.sample_id), row.block)]
        c = cell_index[row.neuron_class]
        if row.bin_index not in range(8):
            raise ValueError("Reviewed observation outside eight-bin window")
        expected[i, k, c, row.bin_index] = row.response
    if (not np.array_equal(np.isfinite(binned), np.isfinite(expected))
            or not np.allclose(binned, expected, rtol=1e-12, atol=1e-14, equal_nan=True)):
        raise ValueError("One-second cache bins or missing support differ from reviewed observations")
    compatibility = fit_representation(raw, baseline, [pair[0] for pair in conditions],
                                        threshold=None, min_animals=2)
    saved_h = pd.read_csv(source / "tables/templates.csv").query('window == "0-40s" and fit_type == "full"')
    saved_h = saved_h.pivot(index="cell", columns="bin_index", values="template").reindex(index=CELLS, columns=range(8))
    saved_a = pd.read_csv(source / "tables/coefficients.csv", dtype={"block": str}).query('window == "0-40s" and fit_type == "full"')
    saved_a = saved_a.pivot(index=["strain", "block"], columns="cell", values="coefficient").reindex(
        index=pd.MultiIndex.from_tuples(conditions, names=["strain", "block"]), columns=CELLS)
    for label, current, saved in [("templates", compatibility["templates"], saved_h.to_numpy()),
                                   ("coefficients", compatibility["coefficients"], saved_a.to_numpy())]:
        if not np.allclose(current, saved, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Unfiltered min_n=2 refit does not reproduce Figure 4 {label}")
    notebook_paths = sorted(notebook_dir.glob("*.ipynb"))
    metadata = dict(input_sha256={str(path): _sha256(path) for path in dict.fromkeys(input_paths)},
                    original_notebook_sha256={str(path): _sha256(path) for path in notebook_paths},
                    reviewed_numeric_sha256=reviewed["numeric_sha256"], raw_source_sha256=old_raw_hash,
                    cache_bin_max_abs_error=float(np.nanmax(np.abs(binned - expected))),
                    old_template_max_abs_error=float(np.nanmax(np.abs(compatibility["templates"] - saved_h.to_numpy()))),
                    old_coefficient_max_abs_error=float(np.nanmax(np.abs(compatibility["coefficients"] - saved_a.to_numpy()))),
                    n_animals=len(animals), n_conditions=len(conditions), n_strains=len(set(p[0] for p in conditions)),
                    n_cells=len(CELLS), complete_animal_curves=int(np.isfinite(raw).all(axis=-1).sum()))
    return dict(raw=raw, baseline_sd=baseline, animals=animals, conditions=conditions,
                cells=CELLS.copy(), metadata=metadata)


def _label(threshold):
    return "unfiltered" if threshold is None else f"{threshold:g}"


def _condition_table(result, conditions, cells):
    rows = []
    for k, (strain, block) in enumerate(conditions):
        for c, cell in enumerate(cells):
            rows.append(dict(strain=strain, block=block, cell=cell,
                             n_animals=int(result["counts"][k, c]),
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


def save_full_representation(data, out, thresholds=(None, .5, 1., 1.5, 2.), primary=1.):
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
        result = fit_representation(data["raw"], data["baseline_sd"], [p[0] for p in conditions], threshold)
        results[label] = result
        all_conditions.append(_condition_table(result, conditions, cells).assign(threshold=label))
        all_templates.append(_template_table(result, cells).assign(threshold=label))
        for c, cell in enumerate(cells):
            status = result["status"][:, c]
            summaries.append(dict(threshold=label, cell=cell,
                                  eligible=int((result["counts"][:, c] >= 3).sum()),
                                  retained=int((status == "retained").sum()),
                                  suppressed=int((status == "below_snr").sum()),
                                  observed_zero=int((status == "observed_zero").sum()),
                                  limited=int((status == "limited_n").sum()), missing=int((status == "missing").sum()),
                                  template_unidentified=int((status == "template_unidentified").sum())))
    main = results[_label(primary)]
    condition_metrics = _condition_table(main, conditions, cells)
    templates = _template_table(main, cells)
    values, strains = aggregate_strains(main["coefficients"], conditions)
    strain_coefficients = pd.DataFrame(values, index=pd.Index(strains, name="strain"), columns=cells)
    sensitivity_metrics = pd.DataFrame(summaries)
    condition_metrics.to_csv(tables / "condition_metrics.csv", index=False)
    templates.to_csv(tables / "templates.csv", index=False)
    strain_coefficients.to_csv(tables / "strain_coefficients.csv")
    sensitivity_metrics.to_csv(tables / "sensitivity_metrics.csv", index=False)
    pd.concat(all_conditions, ignore_index=True).to_csv(tables / "sensitivity_condition_metrics.csv", index=False)
    pd.concat(all_templates, ignore_index=True).to_csv(tables / "sensitivity_templates.csv", index=False)
    np.savez_compressed(tables / "representation_arrays.npz", **{key: value for key, value in main.items()
                        if isinstance(value, np.ndarray)}, cells=np.asarray(cells), conditions=np.asarray(conditions))
    metadata = dict(data["metadata"], primary_threshold=primary,
                    sensitivity_thresholds=list(thresholds), min_animals=3,
                    raw_window_seconds=[0, 40], stimulus_seconds=[0, 10],
                    model_bin_seconds=5, cells=cells, response_units="delta_F_over_F0",
                    gate="P=mean_t(mu^2); V=mean_t(sample variance, ddof=1); C=P-V/n; SNR=sqrt(max(C,0)/V)",
                    zero_scatter="SNR=inf when C>0; SNR=0 when C=0",
                    baseline_snr="RMS condition mean / median animal baseline SD; descriptive auxiliary, not a gate",
                    baseline_noise="Median trial SD over five prestimulus samples; no sqrt(n) rescaling",
                    template_fit="Per cell uncentred weighted SVD of retained condition means; equal strains then equal available dates",
                    template_normalization="RMS=1; greatest-absolute bin positive; first bin breaks an exact tie",
                    thresholds_are="Exploratory settings, not calibrated detection probabilities",
                    coefficient_zero="Adequately observed raw-curve SNR failure, or observed_zero (exactly zero condition mean despite unidentified template); missing/limited_n remains NaN",
                    raw_coefficient="Projection before zeroing onto the threshold-specific fitted template; not an independently fitted unfiltered template",
                    residual_rms="Eight-bin condition mean minus gated reconstruction, in delta_F_over_F0",
                    strain_aggregation="Equal available dates within strain; finite-value mean, missing remains NaN",
                    provenance_verification="One-second bins and support checked against reviewed observations; unfiltered min_n=2 fit checked against old Figure 4",
                    description_only="Full-data gate/templates are descriptive; validation must refit them within training animals",
                    code_sha256=_sha256(__file__))
    (out / "representation_parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dict(primary=main, results=results, condition_metrics=condition_metrics,
                templates=templates, strain_coefficients=strain_coefficients,
                sensitivity_metrics=sensitivity_metrics, metadata=metadata)

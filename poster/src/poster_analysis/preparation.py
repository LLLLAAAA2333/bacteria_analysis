"""Prepare the poster response representation from reviewed animal-level caches.

This is the reviewed individual-SNR algorithm, made independent of report code.
The original pure binning, SNR, weighted-SVD and aggregation routines come from
``analysis/explorations/representation/exploration_response_profiles_20261001/code/response_representation.py``;
the active defaults and table schemas follow ``individual_representation.py``
in the individual-SNR report. No raw fluorescence or source file is modified,
and no calculation or file write runs on import.

Data preparation ends at these full-data descriptive arrays. Split-half and
held-out validation are separate analyses and must refit within each split.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .constants import NEURONS


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


def fit_representation(raw, baseline_sd, strain_ids, threshold=.5, min_animals=2):
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
    return dict(means=means, mean_bins=mean_bins, counts=counts, eligible=eligible, snr=snr,
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
                             raw_residual_rms=result["raw_residual_rms"][k, c],
                             eligible=bool(result["eligible"][k, c])))
    return pd.DataFrame(rows)


def _template_table(result, cells):
    return pd.DataFrame([dict(cell=cell, bin_index=t, time_s=5*t + 2.5,
                              template=result["templates"][c, t],
                              identified=bool(result["template_identified"][c]),
                              n_conditions=int(result["template_n_conditions"][c]))
                         for c, cell in enumerate(cells) for t in range(8)])


def _sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_animal_inputs(curve_path, metrics_path, trial_path, observations_path,
                       cells=NEURONS):
    """Read animal means and validate baseline SD and reviewed five-second bins.

    Parameters are explicit file paths; there is no dependency on a report's
    code or directory layout. ``curve_path`` contains the already-reviewed
    one-second animal-mean ΔF/F₀ curves, obtained by averaging trials per animal.
    ``trial_path`` is read only for five prestimulus samples (-5 through -1 s),
    whose per-trial sample SD and per-animal median must match ``metrics_path``.
    Binned animal curves and their finite/NaN support must exactly match the
    reviewed eight-bin long table at ``observations_path`` within roundoff.

    Return ``raw`` (animal × condition × cell × 40), ``baseline_sd`` (same first
    three axes), ordered ``animals``, ``conditions`` (strain, acquisition block),
    ``cells``, and ``metadata`` containing checks and input content hashes.
    Missing identities remain NaN; every observed curve requires all 40 samples.
    """
    cells = list(cells)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError('Cells must be a nonempty unique ordered list')
    paths = dict(animal_curves=Path(curve_path), animal_metrics=Path(metrics_path),
                 trial_curves=Path(trial_path), observations=Path(observations_path))
    keys = ['sample_id', 'date', 'worm_key', 'neuron_class']
    curves = pd.read_parquet(paths['animal_curves']).reset_index()
    metrics = pd.read_csv(paths['animal_metrics'], dtype={key: str for key in keys})
    for key in keys:
        curves[key] = curves[key].astype(str)
    if curves.duplicated(keys).any() or metrics.duplicated(keys).any():
        raise ValueError('Duplicate animal-condition-cell cache identities')
    indexed = curves.set_index(keys).sort_index()
    metrics = metrics.set_index(keys).sort_index()
    if not indexed.index.equals(metrics.index):
        raise ValueError('Baseline metrics and animal curves have different support')
    if set(curves.neuron_class) != set(cells):
        raise ValueError('Animal curves do not contain exactly the requested cell classes')

    trial_baseline = pd.read_parquet(paths['trial_curves'],
                                     columns=[str(t) for t in range(-5, 0)])
    trial_baseline.index = pd.MultiIndex.from_frame(
        trial_baseline.index.to_frame(index=False).astype({key: str for key in keys}))
    baseline_check = (trial_baseline.std(axis=1, ddof=1)
                      .groupby(level=keys).median().sort_index())
    if (not baseline_check.index.equals(metrics.index)
            or not np.allclose(baseline_check, metrics.baseline_sd,
                               rtol=1e-12, atol=1e-14)):
        raise ValueError('Cached baseline SD differs from median trial prestimulus SD')

    animals = sorted((curves.date + '|' + curves.worm_key).unique())
    conditions = sorted(set(zip(curves.sample_id, curves.date)))
    animal_index = {name: i for i, name in enumerate(animals)}
    condition_index = {pair: k for k, pair in enumerate(conditions)}
    cell_index = {cell: c for c, cell in enumerate(cells)}
    raw = np.full((len(animals), len(conditions), len(cells), 40), np.nan)
    baseline = np.full(raw.shape[:-1], np.nan)
    identities = indexed.index.to_frame(index=False)
    ii = [animal_index[date + '|' + worm]
          for date, worm in zip(identities.date, identities.worm_key)]
    kk = [condition_index[pair] for pair in zip(identities.sample_id, identities.date)]
    cc = [cell_index[cell] for cell in identities.neuron_class]
    raw[ii, kk, cc] = indexed[[str(t) for t in range(40)]].to_numpy(float)
    baseline[ii, kk, cc] = metrics.baseline_sd.to_numpy(float)
    binned = bin_curves(raw)

    observations = pd.read_parquet(paths['observations']).copy()
    observations['date'] = observations.date.astype(str)
    observations['block'] = observations.block.astype(str)
    if not observations.date.eq(observations.block).all():
        raise ValueError('Reviewed acquisition block differs from date')
    observation_keys = ['animal_id', 'sample_id', 'block', 'neuron_class', 'bin_index']
    if observations.duplicated(observation_keys).any():
        raise ValueError('Duplicate reviewed observation features')
    expected = np.full_like(binned, np.nan)
    for row in observations.itertuples(index=False):
        if row.bin_index not in range(8):
            raise ValueError('Reviewed observation outside eight-bin window')
        try:
            i = animal_index[str(row.animal_id)]
            k = condition_index[(str(row.sample_id), row.block)]
            c = cell_index[row.neuron_class]
        except KeyError as exc:
            raise ValueError('Reviewed observation has an unknown animal, condition or cell') from exc
        expected[i, k, c, row.bin_index] = row.response
    if (not np.array_equal(np.isfinite(binned), np.isfinite(expected))
            or not np.allclose(binned, expected, rtol=1e-12, atol=1e-14, equal_nan=True)):
        raise ValueError('One-second cache bins or support differ from reviewed observations')
    metadata = dict(
        input_sha256={name: _sha256(path) for name, path in paths.items()},
        cache_bin_max_abs_error=float(np.nanmax(np.abs(binned - expected))),
        baseline_max_abs_error=float(np.max(np.abs(baseline_check - metrics.baseline_sd))),
        reviewed_bin_support_identical=True,
        n_animals=len(animals), n_conditions=len(conditions),
        n_strains=len({pair[0] for pair in conditions}), n_cells=len(cells),
        complete_animal_curves=int(np.isfinite(raw).all(axis=-1).sum()),
        response_aggregation='Equal trials within animal, then equal animals within strain × block',
        snr_sampling_unit='Animal-mean curve; trial observations are not separate SNR samples',
    )
    return dict(raw=raw, baseline_sd=baseline, animals=animals,
                conditions=conditions, cells=cells, metadata=metadata)


def _strain_audit(condition_metrics):
    """Keep recorded, eligible and suppressed condition support visible."""
    rows = []
    for (strain, cell), group in condition_metrics.groupby(['strain', 'cell'], sort=True):
        rows.append(dict(
            strain=strain, cell=cell, n_dates=len(group),
            n_eligible_dates=int(group.eligible.sum()),
            n_available_dates=int(group.coefficient.notna().sum()),
            n_retained_dates=int(group.status.eq('retained').sum()),
            n_suppressed_dates=int(group.status.eq('below_snr').sum()),
            n_limited_dates=int(group.status.eq('limited_n').sum()),
            n_animals_recorded=int(group.n_animals.sum()),
            coefficient=float(group.coefficient.mean()),
            raw_coefficient=float(group.raw_coefficient.mean()),
            snr_min=float(group.snr.min()), snr_max=float(group.snr.max()),
            date_statuses=';'.join(f'{row.block}:{row.status}'
                                   for row in group.itertuples(index=False)),
        ))
    return pd.DataFrame(rows)


def tables_for_fits(fits, conditions, cells, primary_label='0.5'):
    """Build saved table schemas from explicitly computed threshold-specific fits.

    ``fits`` is an insertion-ordered mapping such as ``{'unfiltered': ...,
    '0.25': ..., '0.5': ..., '0.75': ..., '1': ...}``. Each result is produced by
    ``fit_representation``. Templates are refitted at each threshold upstream;
    this function only assembles tables and never fits or writes files.
    """
    if primary_label not in fits:
        raise ValueError('Primary threshold label must be present in fits')
    cells = list(cells)
    all_conditions, all_templates, summaries = [], [], []
    for label, result in fits.items():
        expected_label = ('unfiltered' if result['threshold'] is None
                          else f"{result['threshold']:g}")
        if label != expected_label:
            raise ValueError(f'Fit label {label!r} differs from its threshold {expected_label!r}')
        if result['counts'].shape != (len(conditions), len(cells)):
            raise ValueError('Fit condition/cell dimensions differ from labels')
        all_conditions.append(_condition_table(result, conditions, cells).assign(threshold=label))
        all_templates.append(_template_table(result, cells).assign(threshold=label))
        for c, cell in enumerate(cells):
            status = result['status'][:, c]
            summaries.append(dict(
                threshold=label, cell=cell,
                eligible=int(result['eligible'][:, c].sum()),
                retained=int((status == 'retained').sum()),
                suppressed=int((status == 'below_snr').sum()),
                observed_zero=int((status == 'observed_zero').sum()),
                limited=int((status == 'limited_n').sum()),
                limited_n=int((status == 'limited_n').sum()),
                missing=int((status == 'missing').sum()),
                template_unidentified=int((status == 'template_unidentified').sum()),
            ))
    primary = fits[primary_label]
    condition_metrics = _condition_table(primary, conditions, cells)
    values, strains = aggregate_strains(primary['coefficients'], conditions)
    coefficients = pd.DataFrame(values, index=pd.Index(strains, name='strain'), columns=cells)
    return dict(
        condition_metrics=condition_metrics, templates=_template_table(primary, cells),
        strain_coefficients=coefficients, strain_audit=_strain_audit(condition_metrics),
        sensitivity_metrics=pd.DataFrame(summaries),
        sensitivity_condition_metrics=pd.concat(all_conditions, ignore_index=True),
        sensitivity_templates=pd.concat(all_templates, ignore_index=True),
    )


def raw_profile_distances(profiles, strains, min_shared=4):
    """Return 1−cosine on complete shared cells and the cell support count.

    The unfiltered profile is strain × cell × eight 5-s bins. A pair is defined
    only with at least ``min_shared`` fully observed cells and norm product above
    1e−12. Missing cells are excluded for that pair, never imputed. This is the
    original descriptive row-order distance, not the chemical comparison RDM.
    """
    values = np.asarray(profiles, dtype=float)
    if (values.ndim != 3 or values.shape[0] != len(strains)
            or np.isinf(values).any() or min_shared < 1):
        raise ValueError('Expected strain × cell × time profiles with finite-or-NaN values')
    complete = np.isfinite(values).all(axis=2)
    shared = complete[:, None, :] & complete[None, :, :]
    n_shared = shared.sum(axis=2)
    numeric = np.nan_to_num(values, nan=0.)
    dot = np.einsum('ict,jct,ijc->ij', numeric, numeric, shared, optimize=True)
    energy = (numeric * numeric).sum(axis=2)
    norm_a = np.einsum('ic,ijc->ij', energy, shared)
    norm_b = np.einsum('jc,ijc->ij', energy, shared)
    denominator = np.sqrt(norm_a * norm_b)
    valid = (n_shared >= min_shared) & (denominator > 1e-12)
    directional = np.divide(dot, denominator, out=np.full(dot.shape, np.nan), where=valid)
    similarity = np.clip((directional + directional.T) / 2, -1., 1.)
    index = pd.Index(strains, name='sample_id')
    return (pd.DataFrame(1 - similarity, index=index, columns=strains),
            pd.DataFrame(n_shared, index=index, columns=strains))


def build_display_tables(primary, conditions, cells):
    """Reconstruct the original raw-profile row order and five-bin display.

    Ordering uses the ungated mean bins where at least ``min_animals`` are
    recorded, averaged equally across available acquisition blocks per strain.
    Finite distances use average linkage; missing distances retain sorted strain
    order rather than imputing. Display values use the primary gated fitted
    reconstruction, aggregated by the same finite-value rule and truncated only
    after fitting all eight bins. Gate-state codes preserve unavailable values.
    """
    cells = list(cells)
    raw_bins = np.where(primary['eligible'][..., None], primary['mean_bins'], np.nan)
    raw_profile, strains = aggregate_strains(raw_bins, conditions)
    rdm, shared = raw_profile_distances(raw_profile, strains)
    raw = rdm.to_numpy()
    if np.isfinite(raw).all():
        distance = np.clip((raw + raw.T) / 2, 0, 2)
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform(distance), method='average'))
        order_rule = 'Average linkage of unfiltered 0–40 s eight-bin cosine distances; display only'
    else:
        order = np.arange(len(strains))
        order_rule = 'Sample ID order; missing unfiltered distances are not imputed'
    ids = [strains[i] for i in order]
    reconstruction, _ = aggregate_strains(primary['reconstruction'], conditions)
    values = reconstruction[order, :, :5].reshape(len(ids), -1)
    display = pd.DataFrame(values, index=pd.Index(ids, name='strain'),
                           columns=[f'{cell}_{5*b}-{5*(b+1)}s' for cell in cells for b in range(5)])
    condition_strains = np.asarray([pair[0] for pair in conditions])
    state_rows = []
    for strain in ids:
        for c, cell in enumerate(cells):
            status = primary['status'][condition_strains == strain, c]
            retained = np.isin(status, ['retained', 'observed_zero'])
            suppressed = status == 'below_snr'
            n_available = int((retained | suppressed).sum())
            if not n_available:
                code, state = np.nan, 'unavailable'
            elif retained.any() and suppressed.any():
                code, state = 0, 'mixed_dates'
            elif retained.any():
                code, state = 1, 'retained'
            else:
                code, state = -1, 'zeroed'
            state_rows.append(dict(
                strain=strain, cell=cell, n_dates=len(status),
                retained_dates=int(retained.sum()), zeroed_dates=int(suppressed.sum()),
                unavailable_dates=len(status) - n_available, state=state, code=code,
            ))
    return dict(rdm_raw=rdm, rdm_raw_shared_cells=shared,
                figure_row_order=pd.DataFrame({'position': range(len(ids)), 'strain': ids}),
                display_profile_5bin=display,
                filter_state_by_strain_cell=pd.DataFrame(state_rows), row_order_rule=order_rule)

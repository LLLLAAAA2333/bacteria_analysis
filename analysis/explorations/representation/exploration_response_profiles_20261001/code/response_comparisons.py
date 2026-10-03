"""Exploratory comparisons of SNR-gated shared-template response profiles.

Splits refit both the gate and templates in globally disjoint animal halves.
LOAO predictions are scored against untouched held-out responses. All output is
descriptive: split repetitions and overlapping RDM pairs are not replicates.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from response_representation import aggregate_strains, bin_curves, fit_representation


METHODS = ("filtered", "unfiltered", "raw")
MIN_SHARED = 4


def _profiles(values):
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or np.isinf(x).any():
        raise ValueError("Expected finite-or-NaN sample × cell [× time] profiles")
    return x


def split_cosine(half_a, half_b, min_shared=MIN_SHARED, support=None):
    """Symmetric cross-half cosine on four-profile complete shared cells.

support optionally restricts sample × cell support further (e.g. identical
support for controls). Undefined directions invalidate their symmetric pair.
Zero norm remains undefined, including a pair of entirely suppressed profiles.
"""
    a, b = _profiles(half_a), _profiles(half_b)
    if a.shape != b.shape or min_shared < 1:
        raise ValueError("Half profiles must have equal shape and positive coverage")
    complete = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)
    if support is not None:
        if np.shape(support) != complete.shape:
            raise ValueError("support must be sample × cell")
        complete &= np.asarray(support, dtype=bool)
    shared = complete[:, None, :] & complete[None, :, :]
    n_shared = shared.sum(axis=2)
    az, bz = np.nan_to_num(a, nan=0.), np.nan_to_num(b, nan=0.)
    dot = np.einsum("ict,jct,ijc->ij", az, bz, shared, optimize=True)
    a_energy, b_energy = (az * az).sum(axis=2), (bz * bz).sum(axis=2)
    norm_a = np.einsum("ic,ijc->ij", a_energy, shared)
    norm_b = np.einsum("jc,ijc->ij", b_energy, shared)
    denominator = np.sqrt(norm_a * norm_b)
    valid = (n_shared >= min_shared) & (denominator > 1e-12)
    directional = np.divide(dot, denominator, out=np.full(dot.shape, np.nan), where=valid)
    symmetric = (directional + directional.T) / 2
    return np.clip(symmetric, -1., 1.), n_shared


def cosine_matrix(profiles, min_shared=MIN_SHARED):
    """Cosine of profiles, preserving undefined zero-norm and missing pairs."""
    return split_cosine(profiles, profiles, min_shared=min_shared)


def _raw_means(raw, min_animals):
    bins = bin_curves(raw)
    counts = np.isfinite(bins).all(axis=-1).sum(axis=0)
    means = np.divide(np.nansum(bins, axis=0), counts[..., None],
                      out=np.full(bins.shape[1:], np.nan), where=counts[..., None] > 0)
    return np.where((counts >= min_animals)[..., None], means, np.nan)


def _aggregate(values, conditions, expected=None):
    result, strains = aggregate_strains(values, conditions)
    if expected is not None and list(strains) != list(expected):
        raise ValueError("Inconsistent strain order across representations")
    return np.asarray(result, float), list(strains)


def _partition_animals(animals, rng):
    """Balance global animal identities within date; never split per strain."""
    dates = np.array([str(animal).split("|", 1)[0] for animal in animals])
    left = np.zeros(len(animals), dtype=bool)
    for date in np.unique(dates):
        positions = rng.permutation(np.flatnonzero(dates == date))
        n_left = len(positions) // 2
        if len(positions) % 2:
            n_left += int(rng.integers(0, 2))
        left[positions[:n_left]] = True
    if not left.any() or left.all():
        raise ValueError("Cannot form two nonempty animal halves")
    return left


def _accumulate(accumulator, values, coverage):
    finite = np.isfinite(values)
    accumulator["sum"] += np.where(finite, values, 0.)
    accumulator["n"] += finite
    accumulator["coverage"] += np.where(finite, coverage, 0)


def _finish(accumulator):
    n = accumulator["n"]
    mean = np.divide(accumulator["sum"], n, out=np.full(n.shape, np.nan), where=n > 0)
    coverage = np.divide(accumulator["coverage"], n, out=np.full(n.shape, np.nan), where=n > 0)
    return mean, n, coverage


def _correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    out = {"n_pairs": int(len(a)), "pearson": None, "spearman": None}
    if len(a) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0:
        out.update(pearson=float(np.corrcoef(a, b)[0, 1]),
                   spearman=float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
    return out


def _matrix_csv(path, matrix, strains):
    pd.DataFrame(matrix, index=pd.Index(strains, name="sample_id"), columns=strains).to_csv(path)


def _finite_mean(x):
    x = np.asarray(x, float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None


def run_split_comparison(data, tables, repeats=100, seed=20261001, progress=None):
    raw, noise = data["raw"], data["baseline_sd"]
    conditions, animals = data["conditions"], data["animals"]
    strain_ids = [str(s) for s, _ in conditions]
    _, strains = _aggregate(_raw_means(raw, 2), conditions)
    size = (len(strains), len(strains))
    def empty():
        return {"sum": np.zeros(size), "n": np.zeros(size, dtype=int), "coverage": np.zeros(size)}
    accumulators = {method: empty() for method in METHODS}
    paired = {method: empty() for method in METHODS}
    rng, partitions, diagnostics = np.random.default_rng(seed), [], []
    for repeat in range(repeats):
        left = _partition_animals(animals, rng)
        for i, animal in enumerate(animals):
            partitions.append({"split": repeat, "animal_id": animal, "half": "A" if left[i] else "B"})
        halves = {}
        for name, mask in (("A", left), ("B", ~left)):
            fits = {"filtered": fit_representation(raw[mask], noise[mask], strain_ids, threshold=1., min_animals=2),
                    "unfiltered": fit_representation(raw[mask], noise[mask], strain_ids, threshold=None, min_animals=2)}
            values = {method: fit["reconstruction"] for method, fit in fits.items()}
            values["raw"] = _raw_means(raw[mask], 2)
            halves[name] = {method: _aggregate(value, conditions, strains)[0] for method, value in values.items()}
            for method, profiles in halves[name].items():
                valid = np.isfinite(profiles).all(axis=2)
                norms = np.nansum(profiles * profiles, axis=(1, 2))
                adequate = valid.sum(axis=1) >= MIN_SHARED
                for j, strain in enumerate(strains):
                    diagnostics.append({"split": repeat, "half": name, "method": method,
                                        "sample_id": strain, "n_complete_cells": int(valid[j].sum()),
                                        "adequate_coverage": bool(adequate[j]),
                                        "all_zero": bool(adequate[j] and norms[j] <= 1e-24)})
        common = np.logical_and.reduce([np.isfinite(halves[half][method]).all(axis=2)
                                        for half in ("A", "B") for method in METHODS])
        comparisons = {}
        for method in METHODS:
            similarity, shared = split_cosine(halves["A"][method], halves["B"][method])
            _accumulate(accumulators[method], similarity, shared)
            comparisons[method] = split_cosine(halves["A"][method], halves["B"][method], support=common)
        valid_all = np.logical_and.reduce([np.isfinite(comparisons[method][0]) for method in METHODS])
        for method, (similarity, shared) in comparisons.items():
            _accumulate(paired[method], np.where(valid_all, similarity, np.nan), shared)
        if progress is not None and ((repeat + 1) % 10 == 0 or repeat == 0):
            progress(f"Independent animal halves: {repeat + 1}/{repeats}")
    facts, paired_values = {}, {}
    for method in METHODS:
        mean, count, shared = _finish(accumulators[method])
        common_mean, common_count, common_shared = _finish(paired[method])
        for field, matrix in (("cosine", mean), ("valid_splits", count), ("shared_cells", shared),
                              ("paired_cosine", common_mean), ("paired_valid_splits", common_count),
                              ("paired_shared_cells", common_shared)):
            _matrix_csv(tables / f"split_{method}_{field}.csv", matrix, strains)
        paired_values[method] = common_mean
        facts[method] = {"diagonal_mean": _finite_mean(np.diag(mean)),
                         "n_valid_diagonal": int(np.isfinite(np.diag(mean)).sum()),
                         "paired_diagonal_mean": _finite_mean(np.diag(common_mean)),
                         "n_valid_paired_diagonal": int(np.isfinite(np.diag(common_mean)).sum())}
    pd.DataFrame(partitions).to_csv(tables / "split_animal_assignments.csv", index=False)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(tables / "split_profile_status.csv", index=False)
    for method in METHODS:
        selected = diag.loc[diag.method.eq(method)]
        n_adequate = int(selected.adequate_coverage.sum())
        n_zero = int(selected.all_zero.sum())
        facts[method].update(n_profile_halves=len(selected), n_adequate_profile_halves=n_adequate,
                             n_all_zero_profile_halves=n_zero,
                             all_zero_fraction_of_adequate=n_zero / n_adequate if n_adequate else None)
    diag.groupby(["method", "sample_id"]).agg(
        n_halves=("split", "size"), n_adequate=("adequate_coverage", "sum"), n_all_zero=("all_zero", "sum")
    ).reset_index().to_csv(tables / "split_profile_status_summary.csv", index=False)
    rows = []
    for i in range(len(strains)):
        for j in range(i, len(strains)):
            rows.append({"strain_a": strains[i], "strain_b": strains[j], "diagonal": i == j,
                         **{method: paired_values[method][i, j] for method in METHODS}})
    pd.DataFrame(rows).to_csv(tables / "split_control_pairs.csv", index=False)
    facts.update(repeats=int(repeats), seed=int(seed), n_strains=len(strains),
                 note="Independent half processing; repeated suppressed zeros do not establish biological repeatability")
    return facts


def heldout_fold(raw, baseline_sd, strain_ids, held_index, threshold=1.):
    """Fit on other animals; held-out targets are never gated or imputed."""
    train = np.arange(len(raw)) != held_index
    noise = baseline_sd[train] if baseline_sd is not None else None
    filtered = fit_representation(raw[train], noise, strain_ids, threshold=threshold, min_animals=2)
    unfiltered = fit_representation(raw[train], noise, strain_ids, threshold=None, min_animals=2)
    actual = bin_curves(raw[held_index])
    mask = (np.isfinite(actual).all(axis=-1)
            & np.isfinite(filtered["reconstruction"]).all(axis=-1)
            & np.isfinite(unfiltered["reconstruction"]).all(axis=-1)
            & (filtered["counts"] >= 2) & (unfiltered["counts"] >= 2))
    return {"filtered_fit": filtered, "unfiltered_fit": unfiltered, "actual": actual, "score_mask": mask}


def _error_summary(frame, keys):
    errors = ["mse_filtered", "mse_unfiltered", "mse_zero"]
    means = frame.groupby(keys, observed=True)[errors].mean().reset_index()
    means["gain_vs_zero"] = means.mse_zero - means.mse_filtered
    means["gain_vs_unfiltered"] = means.mse_unfiltered - means.mse_filtered
    return means


def run_loao_comparison(data, tables, progress=None):
    raw, noise, conditions = data["raw"], data["baseline_sd"], data["conditions"]
    strain_ids = [s for s, _ in conditions]
    rows = []
    for i, animal in enumerate(data["animals"]):
        fold = heldout_fold(raw, noise, strain_ids, i)
        filtered, unfiltered, actual = fold["filtered_fit"], fold["unfiltered_fit"], fold["actual"]
        eligible = np.isfinite(actual).all(axis=-1)
        for k, c in np.argwhere(eligible):
            scored = bool(fold["score_mask"][k, c])
            row = {"animal_id": animal, "sample_id": conditions[k][0], "block": conditions[k][1],
                   "cell": data["cells"][c], "n_train": int(filtered["counts"][k, c]),
                   "train_status": str(filtered["status"][k, c]), "scored": scored,
                   "mse_zero": float(np.mean(actual[k, c] ** 2)) if scored else np.nan,
                   "mse_filtered": float(np.mean((actual[k, c] - filtered["reconstruction"][k, c]) ** 2)) if scored else np.nan,
                   "mse_unfiltered": float(np.mean((actual[k, c] - unfiltered["reconstruction"][k, c]) ** 2)) if scored else np.nan}
            rows.append(row)
        if progress is not None and ((i + 1) % 25 == 0 or i == 0):
            progress(f"Animal-held-out raw-target check: {i + 1}/{len(raw)}")
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "loao_errors.csv", index=False)
    scored = frame.loc[frame.scored].copy()
    if scored.empty:
        return {"n_scored": 0, "n_observed": len(frame)}
    condition = _error_summary(scored, ["sample_id", "block", "cell"])
    condition = condition.merge(scored.groupby(["sample_id", "block", "cell"]).size().rename("n_scored_animals").reset_index())
    condition.to_csv(tables / "loao_condition_summary.csv", index=False)
    # Avoid giving strains with more blocks or animals larger cell-summary weight.
    strain = _error_summary(condition, ["sample_id", "cell"])
    cell = _error_summary(strain, ["cell"])
    cell.to_csv(tables / "loao_cell_summary.csv", index=False)
    _error_summary(scored, ["animal_id", "cell"]).to_csv(tables / "loao_animal_summary.csv", index=False)
    status_condition = _error_summary(scored, ["sample_id", "block", "cell", "train_status"])
    status_strain = _error_summary(status_condition, ["sample_id", "cell", "train_status"])
    status = _error_summary(status_strain, ["cell", "train_status"])
    status = status.merge(scored.groupby(["cell", "train_status"]).size().rename("n_scored_observations").reset_index())
    status.to_csv(tables / "loao_status_summary.csv", index=False)
    overall = {column: float(cell[column].mean()) for column in
               ("mse_filtered", "mse_unfiltered", "mse_zero", "gain_vs_zero", "gain_vs_unfiltered")}
    return {"n_scored": len(scored), "n_observed": len(frame), "n_animals_scored": int(scored.animal_id.nunique()),
            "overall_equal_cells_strains_blocks": overall,
            "note": "Untouched held-out raw-bin targets; common finite support for gated, unfiltered and zero predictions"}


def run_rdm_comparison(data, full_fit, tables, reports_dir, sensitivity=None, fits=None):
    conditions, cells = data["conditions"], data["cells"]
    raw, noise = data["raw"], data["baseline_sd"]
    coefficients, strains = _aggregate(full_fit["coefficients"], conditions)
    unfiltered = fit_representation(raw, noise, [s for s, _ in conditions], threshold=None, min_animals=3)
    vectors = {"filtered": coefficients,
               "unfiltered": _aggregate(unfiltered["coefficients"], conditions, strains)[0],
               "raw": _aggregate(_raw_means(raw, 3), conditions, strains)[0]}
    if sensitivity:
        for name, value in sensitivity.items():
            if np.shape(value)[0] != len(strains):
                raise ValueError("Sensitivity matrices must already use the primary strain order")
            vectors[name] = value
    if fits:
        for name, fit in fits.items():
            if name in vectors:
                raise ValueError(f"Sensitivity label conflicts with primary representation: {name}")
            vectors[name] = _aggregate(fit["coefficients"], conditions, strains)[0]
    chemical_path = Path(reports_dir) / "population_first_20260930/tables/aligned_chemical_log2fc_paired.csv"
    chemical = pd.read_csv(chemical_path, index_col=0)
    if not chemical.index.is_unique or not set(strains) <= set(chemical.index):
        raise ValueError("Cached chemical matrix does not cover unique neural strain IDs")
    chemical = chemical.loc[strains]
    values = chemical.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Chemical log2FC cache contains nonfinite entries")
    chemical_rdm = np.sqrt(np.mean((values[:, None] - values[None, :]) ** 2, axis=2))
    _matrix_csv(tables / "rdm_chemical.csv", chemical_rdm, strains)
    rdms = {}
    for method, vector in vectors.items():
        similarity, shared = cosine_matrix(vector)
        rdms[method] = 1 - similarity
        _matrix_csv(tables / f"rdm_{method}.csv", rdms[method], strains)
        _matrix_csv(tables / f"rdm_{method}_shared_cells.csv", shared, strains)
    i, j = np.triu_indices(len(strains), 1)
    paired = pd.DataFrame({"strain_a": np.asarray(strains)[i], "strain_b": np.asarray(strains)[j],
                           "chemical": chemical_rdm[i, j], **{name: value[i, j] for name, value in rdms.items()}})
    common = np.isfinite(paired[list(METHODS)]).all(axis=1)
    common_all = np.isfinite(paired[list(rdms)]).all(axis=1)
    paired["common_primary_controls"] = common
    paired["common_all_representations"] = common_all
    paired.to_csv(tables / "rdm_matched_pairs.csv", index=False)
    records = []
    for name in rdms:
        records.append({"comparison": f"{name}_vs_chemical", "support": "own_valid_pairs",
                        **_correlations(paired[name], paired.chemical)})
        records.append({"comparison": f"{name}_vs_chemical", "support": "common_primary_controls",
                        **_correlations(paired.loc[common, name], paired.loc[common, "chemical"])})
        records.append({"comparison": f"{name}_vs_chemical", "support": "common_all_representations",
                        **_correlations(paired.loc[common_all, name], paired.loc[common_all, "chemical"])})
        if name != "filtered":
            records.append({"comparison": f"filtered_vs_{name}", "support": "common_primary_controls",
                            **_correlations(paired.loc[common, "filtered"], paired.loc[common, name])})
            records.append({"comparison": f"filtered_vs_{name}", "support": "common_all_representations",
                            **_correlations(paired.loc[common_all, "filtered"], paired.loc[common_all, name])})
    pd.DataFrame(records).to_csv(tables / "rdm_comparison_summary.csv", index=False)
    return {"n_strains": len(strains), "n_cells": len(cells), "n_chemical_features": chemical.shape[1],
            "n_possible_pairs": len(paired), "n_common_primary_control_pairs": int(common.sum()),
            "n_common_all_representation_pairs": int(common_all.sum()),
            "correlations": records, "chemical_source": str(chemical_path),
            "chemical_sha256": hashlib.sha256(chemical_path.read_bytes()).hexdigest(),
            "note": "Descriptive correlations only; pairs share strains and are not independent replicates"}


def run_comparisons(data, full_fit, output_dir, reports_dir, repeats=100, seed=20261001,
                    sensitivity=None, progress=print, fits=None):
    """Write comparison tables only; original data, notebooks and reports untouched."""
    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    return {"split": run_split_comparison(data, tables, repeats, seed, progress),
            "loao": run_loao_comparison(data, tables, progress),
            "rdm": run_rdm_comparison(data, full_fit, tables, reports_dir, sensitivity, fits)}

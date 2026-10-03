"""Animal-held-out, uncentred shared-template calcium response models.

Arrays are animal × (strain, acquisition block) × neuron class × 5-second bin.
All responses retain the input ΔF/F0 scale. NaNs are unobserved, never responses.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("B", "M0", "M1", "M2")
SEED = 20260930
TOL = 1e-10
MAX_ITER = 2000


def assessment_fingerprint(summary):
    """Bind manually reviewed judgments to their numerical model/coverage table."""
    columns = ["window", "cell", "mse_B", "mse_M0", "mse_M1", "mse_M2", "mse_M1_top3",
               "n_animals", "n_strains", "n_conditions"]
    canonical = summary[columns].sort_values(["window", "cell"]).to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(canonical.encode()).hexdigest()


def evidence_fingerprints(output_dir):
    """Bind judgments to curves and settings; canonicalize preparation metadata.

    Provenance-only additions such as a verification log must not invalidate an
    identical rerun. Scientific preparation fields and the raw hash are bound.
    """
    root = Path(output_dir)
    names = ["tables/predictions.parquet", "tables/templates.csv", "tables/coefficients.csv",
             "tables/template_stability.csv", "tables/top3_exclusions.csv",
             "analysis_parameters.json"]
    result = {}
    for name in names:
        with (root/name).open("rb") as handle:
            result[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = json.loads((root/"data/preparation_metadata.json").read_text())
    keys = ["raw_sha256", "aggregation", "classes", "bilateral_pooling", "excluded_neurons",
            "animal_identity", "block_definition", "onset_index", "offset_index_exclusive",
            "volume_interval_seconds", "window_seconds", "bin_seconds", "source_time_points",
            "response_units", "missingness"]
    canonical = json.dumps({key: metadata.get(key) for key in keys}, sort_keys=True)
    result["data/preparation_metadata.json"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def feature_weights(mask, strain_ids):
    """Equal cells, strains within cell, blocks within strain, bins within block.

These weights apply to condition means. Each underlying animal has weight
feature_weight / n_animals at that feature. Missing features have zero weight.
"""
    mask = np.asarray(mask, dtype=bool)
    strains = np.asarray(strain_ids)
    if mask.ndim != 3 or len(strains) != mask.shape[0]:
        raise ValueError("Expected condition × cell × bin mask and condition strain IDs")
    w = np.zeros(mask.shape, dtype=float)
    active_cells = np.flatnonzero(mask.any(axis=(0, 2)))
    for c in active_cells:
        present = mask[:, c].any(axis=1)
        unique = np.unique(strains[present])
        for s in unique:
            rows = np.flatnonzero(present & (strains == s))
            for k in rows:
                w[k, c, mask[k, c]] = 1 / (
                    len(active_cells) * len(unique) * len(rows) * mask[k, c].sum()
                )
    return w


def _divide(num, den, fill=0.):
    return np.divide(num, den, out=np.full(np.shape(num), fill, dtype=float), where=den > 0)


def _connected_support(valid):
    """All observed columns must be linked by rows with shared observations."""
    active = np.flatnonzero(valid.any(axis=0))
    if not len(active):
        return False
    v = valid[:, active].astype(int)
    adjacency = (v.T @ v) > 0
    reached = np.zeros(len(active), dtype=bool)
    reached[0] = True
    while True:
        following = reached | adjacency[reached].any(axis=0)
        if np.array_equal(following, reached):
            return bool(reached.all())
        reached = following


def _rank_one(y, w, nonnegative=False, initial=None, seed=SEED):
    """Weighted alternating LS; nonnegative row scores only for M0.

Zero-valued work buffers below implement masked arithmetic, not imputation.
Unsupported coordinates are restored to NaN. Fully observed M1 uses exact SVD.
"""
    valid = (w > 0) & np.isfinite(y)
    w = np.where(valid, w, 0.)
    z = np.where(valid, y, 0.)
    rows, cols = w.sum(axis=1) > 0, w.sum(axis=0) > 0
    if not valid.any():
        return np.full(y.shape[0], np.nan), np.full(y.shape[1], np.nan), np.full_like(y, np.nan), {"identified": False, "converged": True, "iterations": 0}
    if np.sum(w * z * z) == 0:
        return np.where(rows, 0., np.nan), np.full(y.shape[1], np.nan), np.where(valid, 0., np.nan), {"identified": False, "converged": True, "iterations": 0}
    yr, wr = z[np.ix_(rows, cols)], w[np.ix_(rows, cols)]
    complete_row_weights = (wr > 0).all() and np.allclose(wr, wr[:, :1], rtol=1e-12, atol=0)
    if not nonnegative and complete_row_weights:
        _, _, vt = np.linalg.svd(np.sqrt(wr[:, :1]) * yr, full_matrices=False)
        h = np.zeros(y.shape[1])
        h[cols] = vt[0]
        u = _divide(np.sum(w * z * h, axis=1), np.sum(w * h * h, axis=1))
        converged, iterations = True, 1
    else:
        rng = np.random.default_rng(seed)
        starts = [_divide(np.sum(w * z, axis=0), w.sum(axis=0))]
        if initial is not None:
            starts.insert(0, np.nan_to_num(initial, nan=0.))
        # SVD and observed high-energy rows initialize ALS; missing entries still
        # have exactly zero weight in every update and in model selection.
        _, _, vt = np.linalg.svd(np.sqrt(w) * z, full_matrices=False)
        starts.extend([vt[0], -vt[0]])
        for k in np.argsort(np.sum(w * z * z, axis=1))[-2:]:
            starts.extend([z[k], -z[k]])
        starts.extend(rng.normal(size=y.shape[1]) for _ in range(3))
        best = None
        for start in starts:
            h0 = np.where(cols, start, 0.).astype(float)
            if np.linalg.norm(h0) == 0:
                continue
            h0 /= np.linalg.norm(h0)
            previous = np.inf
            for it in range(MAX_ITER):
                u0 = _divide(np.sum(w * z * h0, axis=1), np.sum(w * h0 * h0, axis=1))
                if nonnegative:
                    u0 = np.maximum(u0, 0.)
                hn = _divide(np.sum(w * z * u0[:, None], axis=0), np.sum(w * u0[:, None] ** 2, axis=0))
                norm = np.linalg.norm(hn)
                if norm > 0:
                    hn /= norm
                    u0 *= norm
                loss = np.sum(w * (z - u0[:, None] * hn) ** 2)
                h0 = hn
                converged0 = np.isfinite(previous) and abs(previous - loss) <= TOL * max(previous, 1e-14)
                if converged0:
                    break
                previous = loss
            if best is None or loss < best[0]:
                best = (loss, u0.copy(), h0.copy(), converged0, it + 1)
        _, u, h, converged, iterations = best
    rms = np.sqrt(np.mean(h[cols] ** 2))
    if rms <= 1e-14:
        return np.where(rows, 0., np.nan), np.full(y.shape[1], np.nan), np.where(valid, 0., np.nan), {"identified": False, "converged": converged, "iterations": iterations}
    h /= rms
    u *= rms
    if not nonnegative and h[np.argmax(abs(h))] < 0:
        h, u = -h, -u
    prediction = np.where(valid, u[:, None] * h, np.nan)
    h[~cols], u[~rows] = np.nan, np.nan
    return u, h, prediction, {"identified": _connected_support(valid), "converged": converged, "iterations": iterations}


def fit_models(means, counts, strain_ids, min_train=2, seed=SEED):
    """Fit all four nested models on identical training means and support."""
    y = np.asarray(means, dtype=float)
    mask = np.isfinite(y) & (np.asarray(counts) >= min_train)
    w = feature_weights(mask, strain_ids)
    z = np.where(mask, y, 0.)
    baseline = _divide((w * z).sum(axis=0), w.sum(axis=0), fill=np.nan)
    k, nc, nt = y.shape
    g, flat_h, m0, d0 = _rank_one(y.reshape(k, -1), w.reshape(k, -1), True, seed=seed)
    population = flat_h.reshape(nc, nt)
    templates = np.full((nc, nt), np.nan)
    coefficients = np.full((k, nc), np.nan)
    m1 = np.full_like(y, np.nan)
    diagnostics = [dict(model="M0", cell_index=-1, **d0)]
    for c in range(nc):
        a, h, p, d = _rank_one(y[:, c], w[:, c], initial=population[c], seed=seed+c)
        templates[c], coefficients[:, c], m1[:, c] = h, a, p
        diagnostics.append(dict(model="M1", cell_index=c, **d))
    predictions = {"B": np.where(mask, baseline[None], np.nan), "M0": m0.reshape(y.shape),
                   "M1": m1, "M2": np.where(mask, y, np.nan)}
    losses = {m: float(np.sum(w[mask] * (y[mask] - p[mask]) ** 2)) for m, p in predictions.items()}
    tolerance = 1e-8 * max(losses["B"], 1.)
    if any(losses[a] + tolerance < losses[b] for a, b in zip(MODELS[:-1], MODELS[1:])):
        raise RuntimeError(f"Nested training loss check failed: {losses}")
    return dict(predictions=predictions, templates=templates, coefficients=coefficients,
                gains=g, population_template=population, baseline=baseline, mask=mask,
                weights=w, losses=losses, diagnostics=diagnostics)


def _top_three_refit(means, counts, strain_ids, original, seed=SEED):
    """Exclude three training-selected strains only from template estimation."""
    k, nc, nt = means.shape
    strains = np.asarray(strain_ids)
    mask = original["mask"]
    pred = np.full_like(means, np.nan)
    templates = np.full((nc, nt), np.nan)
    records = []
    for c in range(nc):
        scores = {}
        for s in np.unique(strains[mask[:, c].any(axis=1)]):
            rows = np.flatnonzero((strains == s) & mask[:, c].any(axis=1))
            scores[s] = np.sqrt(np.mean([np.mean(means[j, c, mask[j, c]] ** 2) for j in rows]))
        ranked = sorted(scores, key=lambda s: (-scores[s], s))
        excluded = ranked[:3]
        reduced = mask.copy()
        reduced[np.isin(strains, excluded), c] = False
        w = feature_weights(reduced, strain_ids)
        _, h, _, diag = _rank_one(means[:, c], w[:, c], seed=seed+c)
        # A template cannot support a full-window stress prediction when its
        # remaining training support has disconnected or missing time bins.
        identified = bool(diag["identified"] and np.isfinite(h).all())
        if identified:
            z = np.where(mask[:, c], means[:, c], 0.)
            ow = original["weights"][:, c]
            a = _divide((ow*z*h).sum(axis=1), (ow*h*h).sum(axis=1), fill=np.nan)
            pred[:, c] = np.where(mask[:, c], a[:, None]*h, np.nan)
            templates[c] = h
        for rank, s in enumerate(excluded, 1):
            records.append(dict(cell_index=c, sample_id=s, rank=rank, training_rms=float(scores[s]),
                                remaining_strains=len(ranked)-len(excluded), identified=identified))
    return pred, templates, records


def leave_one_animal_out(values, strain_ids, stress=True, progress=None):
    """Remove an animal's complete slab before all fitting and predict it once."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 4:
        raise ValueError("Expected animal × condition × cell × bin")
    n, k, nc, nt = x.shape
    counts = np.isfinite(x).sum(axis=0)
    score_mask = np.isfinite(x) & (counts[None] >= 3)
    predictions = {m: np.full_like(x, np.nan) for m in MODELS + (("M1_top3",) if stress else ())}
    templates = np.full((n, nc, nt), np.nan)
    stress_templates = np.full_like(templates, np.nan)
    coefficients = np.full((n, k, nc), np.nan)
    diagnostics, excluded = [], []
    for i in range(n):
        # Directly slice training animals rather than subtract held-out values;
        # even floating-point cancellation cannot depend on test amplitudes.
        train = x[np.arange(n) != i]
        num = np.isfinite(train).sum(axis=0)
        mean = _divide(np.nansum(train, axis=0), num, fill=np.nan)
        fit = fit_models(mean, num, strain_ids)
        for m in MODELS:
            predictions[m][i] = np.where(score_mask[i], fit["predictions"][m], np.nan)
        templates[i], coefficients[i] = fit["templates"], fit["coefficients"]
        for record in fit["diagnostics"]:
            diagnostics.append(dict(fold_index=i, mean_target_loss=fit["losses"][record["model"]], **record))
        if stress:
            pred, h, rec = _top_three_refit(mean, num, strain_ids, fit)
            predictions["M1_top3"][i] = np.where(score_mask[i], pred, np.nan)
            stress_templates[i] = h
            excluded.extend(dict(fold_index=i, **r) for r in rec)
        if progress is not None:
            progress(i+1, n)
    return dict(predictions=predictions, templates=templates, stress_templates=stress_templates,
                coefficients=coefficients, score_mask=score_mask, counts=counts,
                diagnostics=diagnostics, exclusions=excluded)


def to_tensor(observations, classes, n_bins=8):
    """Return values and explicit axis labels; missing classes stay NaN."""
    obs = observations.loc[observations.bin_index < n_bins]
    animals = sorted(obs.animal_id.unique())
    conditions = list(obs[["sample_id", "block"]].drop_duplicates().sort_values(["sample_id", "block"]).itertuples(index=False, name=None))
    aidx, kidx, cidx = ({v:i for i,v in enumerate(axis)} for axis in (animals, conditions, classes))
    values = np.full((len(animals), len(conditions), len(classes), n_bins), np.nan)
    key = ["animal_id", "sample_id", "block", "neuron_class", "bin_index"]
    if obs.duplicated(key).any():
        raise ValueError("Duplicate animal-condition-cell-bin observations")
    for r in obs.itertuples(index=False):
        values[aidx[r.animal_id], kidx[(r.sample_id, r.block)], cidx[r.neuron_class], r.bin_index] = r.response
    return values, animals, conditions


def _add_gains(table):
    table = table.copy()
    table["delta_combination"] = table.mse_M0 - table.mse_M1
    table["delta_timing"] = table.mse_M1 - table.mse_M2
    table["delta_B_M1"] = table.mse_B - table.mse_M1
    table["delta_B_M2"] = table.mse_B - table.mse_M2
    if "mse_M1_top3" in table:
        table["delta_B_M1_top3"] = table.mse_B - table.mse_M1_top3
        table["delta_top3_penalty"] = table.mse_M1_top3 - table.mse_M1
    return table


def summarize_predictions(predictions):
    """Aggregate bins/blocks/strains/cells without treating them as replicates."""
    scored = predictions.loc[predictions.eligible].copy()
    models = list(MODELS) + (["M1_top3"] if "pred_M1_top3" in scored else [])
    for m in models:
        scored[f"mse_{m}"] = (scored.actual - scored[f"pred_{m}"]) ** 2
    errors = [f"mse_{m}" for m in models]
    feature_keys = ["window", "strain", "block", "cell", "bin_index"]
    features = scored.groupby(feature_keys, observed=True)[errors].mean().reset_index()
    # Stress comparisons must never silently average only successful fits.
    if "M1_top3" in models:
        complete = scored.groupby(feature_keys).pred_M1_top3.count().eq(scored.groupby(feature_keys).size())
        features.loc[~pd.MultiIndex.from_frame(features[feature_keys]).map(complete).to_numpy(), "mse_M1_top3"] = np.nan
    condition_keys = ["window", "strain", "block", "cell"]
    condition = features.groupby(condition_keys)[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index()
    cond_n = scored.groupby(condition_keys).agg(n_animals=("animal_id", "nunique"), n_bins=("bin_index", "nunique")).reset_index()
    condition = _add_gains(condition.merge(cond_n, on=condition_keys))
    strain = condition.groupby(["window", "strain", "cell"])[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index()
    cell = strain.groupby(["window", "cell"])[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index()
    c_n = scored.groupby(["window", "cell"]).agg(n_animals=("animal_id", "nunique"), n_strains=("strain", "nunique"), n_entries=("actual", "size")).reset_index()
    k_n = condition.groupby(["window", "cell"]).size().rename("n_conditions").reset_index()
    cell = _add_gains(cell.merge(c_n).merge(k_n))
    cell["status"] = "pending"
    for m in models:
        cell[f"rmse_{m}"] = np.sqrt(cell[f"mse_{m}"])
    overall = _add_gains(cell.groupby("window")[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index())
    overall["n_cells"] = overall.window.map(cell.groupby("window").size())
    # Every animal-condition-cell curve is retained, before population weighting.
    animal_keys = ["window", "animal_id", "strain", "block", "cell"]
    animal = scored.groupby(animal_keys)[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index()
    animal = _add_gains(animal.merge(scored.groupby(animal_keys).size().rename("n_bins").reset_index()))
    animal_cell = _add_gains(animal.groupby(["window", "animal_id", "cell"])[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index())
    stages = {0:"0-10s", 1:"0-10s", 2:"10-25s", 3:"10-25s", 4:"10-25s", 5:"25-40s", 6:"25-40s", 7:"25-40s"}
    features["stage"] = features.bin_index.map(stages)
    stage = _add_gains(features.groupby(condition_keys+["stage"])[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index())
    scored["stage"] = scored.bin_index.map(stages)
    animal_stage = _add_gains(scored.groupby(animal_keys+["stage"])[errors].agg(lambda x: x.mean() if x.notna().all() else np.nan).reset_index())
    for keys, detail, target in ((condition_keys, animal, condition), (condition_keys+["stage"], animal_stage, stage)):
        evidence = detail.assign(positive_timing=detail.delta_timing > 0, positive_B_M2=detail.delta_B_M2 > 0,
                                 positive_B_M1=detail.delta_B_M1 > 0,
                                 joint_timing_and_B=(detail.delta_timing > 0) & (detail.delta_B_M2 > 0)).groupby(keys).agg(
            n_animals_positive_timing=("positive_timing", "sum"),
            n_animals_positive_B_M2=("positive_B_M2", "sum"),
            n_animals_positive_B_M1=("positive_B_M1", "sum"),
            n_animals_joint_timing_and_B=("joint_timing_and_B", "sum"),
            n_animals_evaluated=("animal_id", "nunique")).reset_index()
        merged = target.merge(evidence, on=keys)
        if target is condition:
            condition = merged
        else:
            stage = merged
    return dict(cell_summary=cell, condition_summary=condition, strain_summary=_add_gains(strain),
                overall_summary=overall, animal_errors=animal, animal_cell_errors=animal_cell,
                stage_summary=stage, animal_stage_errors=animal_stage, feature_errors=features)


def run_analysis(observations, classes, output_dir, progress=print):
    """Run exactly 40 s main, 25 s sensitivity, and main-window top-three check.

    This explicit Notebook-callable function writes derived tables only.
    It does not classify neurons automatically or modify the source parquet.
    """
    out = Path(output_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    all_predictions, all_templates, all_coefficients, all_coverage = [], [], [], []
    all_diagnostics, all_exclusions, all_fold_coefficients, all_stability = [], [], [], []
    fold_records = []
    for nb in (8, 5):
        window = f"0-{nb*5}s"
        x, animals, conditions = to_tensor(observations, classes, nb)
        strain_ids = [s for s, _ in conditions]
        def notify(i, n):
            if progress is not None and (i == 1 or i % 10 == 0 or i == n):
                progress(f"{window}: animal {i}/{n}")
        cv = leave_one_animal_out(x, strain_ids, stress=(nb==8), progress=notify)
        n = cv["counts"]
        mean = _divide(np.nansum(x, axis=0), n, fill=np.nan)
        full = fit_models(mean, n, strain_ids)
        full_identified = {d["cell_index"]: d["identified"] for d in full["diagnostics"] if d["model"] == "M1"}
        fold_identified = {(d["fold_index"], d["cell_index"]): d["identified"] for d in cv["diagnostics"] if d["model"] == "M1"}
        for m in MODELS:
            if not np.array_equal(np.isfinite(cv["predictions"][m]), cv["score_mask"]):
                raise RuntimeError(f"Model-specific OOF mask: {window} {m}")
        for i, animal in enumerate(animals):
            fold_records.append(dict(window=window, heldout_animal=animal,
                                     n_train_animals=len(animals)-1,
                                     training_animals=json.dumps([a for a in animals if a != animal])))
        # Include unscored observations so raw evidence remains visible.
        for i,k,c,t in np.argwhere(np.isfinite(x)):
            s,b = conditions[k]
            record = dict(window=window, animal_id=animals[i], strain=s, block=b, cell=classes[c],
                          bin_index=int(t), time_s=float(t*5+2.5), actual=x[i,k,c,t],
                          eligible=bool(cv["score_mask"][i,k,c,t]), n_train=int(n[k,c,t]-1), n_animals=int(n[k,c,t]))
            record.update({f"pred_{m}": p[i,k,c,t] for m,p in cv["predictions"].items()})
            record["unscored_reason"] = "" if record["eligible"] else "fewer_than_two_training_animals"
            all_predictions.append(record)
        for k,(s,b) in enumerate(conditions):
            for c,cell in enumerate(classes):
                all_coefficients.append(dict(window=window, strain=s, block=b, cell=cell,
                                             coefficient=full["coefficients"][k,c], gain_M0=full["gains"][k],
                                             fit_type="full", n_animals_min=int(n[k,c].min()), n_animals_max=int(n[k,c].max())))
                for t in range(nb):
                    all_coverage.append(dict(window=window, strain=s, block=b, cell=cell, bin_index=t,
                        time_s=t*5+2.5, n_animals=int(n[k,c,t]), n_scored_animals=int(n[k,c,t]) if n[k,c,t]>=3 else 0))
                for i,a in enumerate(animals):
                    all_fold_coefficients.append(dict(window=window, fold=a, strain=s, block=b, cell=cell,
                                                      coefficient=cv["coefficients"][i,k,c]))
        for c,cell in enumerate(classes):
            hfull = full["templates"][c]
            for t in range(nb):
                all_templates.append(dict(window=window, cell=cell, bin_index=t, time_s=t*5+2.5,
                    template=hfull[t], fit_type="full", fold="full",
                    identified=full_identified[c],
                    population_template=full["population_template"][c,t], baseline=full["baseline"][c,t]))
            for i,a in enumerate(animals):
                h = cv["templates"][i,c]
                hs = cv["stress_templates"][i,c]
                informative = bool((np.isfinite(x[i,:,c]) & (n[:,c] >= 2)).any())
                identified = full_identified[c] and fold_identified[i,c]
                def cosine(u,v):
                    return abs(float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))) if np.isfinite(u).all() and np.isfinite(v).all() and np.linalg.norm(u)*np.linalg.norm(v)>0 else np.nan
                all_stability.append(dict(window=window, cell=cell, fold=a, informative_fold=informative,
                                          identified=identified,
                                          abs_cosine_to_full=cosine(h,hfull) if identified else np.nan,
                                          abs_cosine_top3=cosine(h,hs) if fold_identified[i,c] else np.nan))
                for fit_type, hh in [("loao",h)] + ([("top3_loao",hs)] if nb==8 else []):
                    for t in range(nb):
                        all_templates.append(dict(window=window, cell=cell, bin_index=t, time_s=t*5+2.5,
                                                  template=hh[t], fit_type=fit_type, fold=a, informative_fold=informative,
                                                  identified=fold_identified[i,c] if fit_type == "loao" else bool(np.isfinite(hh).all())))
        for d in cv["diagnostics"]:
            all_diagnostics.append(dict(window=window, fold=animals[d["fold_index"]], **d))
        for d in full["diagnostics"]:
            all_diagnostics.append(dict(window=window, fold="full", mean_target_loss=full["losses"][d["model"]], **d))
        for e in cv["exclusions"]:
            all_exclusions.append(dict(window=window, fold=animals[e["fold_index"]], cell=classes[e["cell_index"]], **e))
    pred = pd.DataFrame(all_predictions)
    pred.to_parquet(out / "predictions.parquet", index=False)
    tables = summarize_predictions(pred)
    tables.update(templates=pd.DataFrame(all_templates), coefficients=pd.DataFrame(all_coefficients),
                  coverage=pd.DataFrame(all_coverage), fit_diagnostics=pd.DataFrame(all_diagnostics),
                  top3_exclusions=pd.DataFrame(all_exclusions), fold_coefficients=pd.DataFrame(all_fold_coefficients),
                  template_stability=pd.DataFrame(all_stability), folds=pd.DataFrame(fold_records))
    for name,table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    parameters = dict(seed=SEED, tolerance=TOL, max_iterations=MAX_ITER, windows_seconds=[40,25],
        stimulus_seconds=[0,10], min_training_animals_per_window=2, block="date (recorded acquisition proxy)",
        weights="equal cells; strains within cell; blocks within strain; bins within condition; animals within feature",
        main_loss="MSE in squared original delta_F_over_F0 units; negative gains retained", cell_scaling="none",
        uncertainty="descriptive paired OOF animal results; overlapping folds not independent; no inferential CIs",
        top3_rule="40s only: per-cell training means RMS across bins, then squared RMS averaged equally across blocks per strain; largest 3 strains; ID breaks ties; exclude all their blocks only from template fit; refit coefficients on all original training means",
        template_normalization="RMS=1; maximum absolute bin positive, earliest breaks ties; M0 positive scale only",
        baseline_definition="input delta_F_over_F0 retained; upstream 01 notebook documents baseline fit/AIC",
        classes=list(classes), fit_algorithm="row-weighted complete M1: exact uncentred SVD; otherwise masked weighted alternating LS with multiple starts",
        training_mean_loss_note="diagnostic mean-target residual excludes animal scatter constant; used only for solver/nesting checks")
    (Path(output_dir)/"analysis_parameters.json").write_text(json.dumps(parameters,indent=2,ensure_ascii=False),encoding="utf-8")
    return tables

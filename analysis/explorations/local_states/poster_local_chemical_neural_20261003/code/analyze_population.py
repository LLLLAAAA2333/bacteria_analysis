"""Notebook-callable bounded chemical-state fit to all 13 unit coordinates.

Call run_analysis(repo_root, out) once, or pass overwrite=True for an explicit
rerun. No raw files or existing reports are changed. See ../PROTOCOL.md for the
predeclared scope. All full-data associations and rank groups are descriptive.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform

import numpy as np
import pandas as pd
import scipy

NEURONS = ["ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
           "ASEL", "ASER", "AWCON", "AWCOFF"]
SCHEMES = ["leave_one_strain_out", "leave_one_recorded_species_out"]
SELECTION = "minimum summed training SSE over 13 unscaled unit coordinates; exact ties by sorted member names"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_axes_api(repo_root):
    path = Path(repo_root) / "reports/exploration_bacteroides_local_model_20261003/chemical/code/local_chemical_axes.py"
    spec = importlib.util.spec_from_file_location("unchanged_local_chemical_axes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def fit_ols(x, y):
    """OLS of n x 13 response on intercept and one chemical score, unscaled."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xm, ym = float(x.mean()), y.mean(axis=0)
    dx, dy = x - xm, y - ym
    xx = float(dx @ dx)
    slope = dx @ dy / xx if xx > 1e-24 else np.zeros(y.shape[1])
    intercept = ym - slope * xm
    prediction = intercept + x[:, None] * slope
    sse = ((y - prediction) ** 2).sum(axis=0)
    null_sse = (dy ** 2).sum(axis=0)
    denom = np.sqrt(xx * null_sse)
    r = np.divide(dx @ dy, denom, out=np.zeros(y.shape[1]), where=denom > 1e-24)
    return {"intercept": intercept.tolist(), "slope": slope.tolist(),
            "training_mean": ym.tolist(), "training_predictor_mean": xm,
            "sse_model": sse.tolist(), "sse_mean": null_sse.tolist(),
            "sse_model_total": float(sse.sum()), "sse_mean_total": float(null_sse.sum()),
            "pearson_r": r.tolist(), "predictor_constant": bool(xx <= 1e-24)}


def fit_selected_state(x, metadata, y, axes_api):
    """Refit all chemical operations and select by all-coordinate training SSE."""
    if not x.index.equals(y.index):
        raise ValueError("Chemical and neural strain orders must agree")
    axes = axes_api.fit_axes(x, metadata)
    candidates = []
    for state, members in axes["module_members"].items():
        model = fit_ols(axes["train_scores"][state], y)
        candidates.append({"state_id": state, "members": members,
                           "n_annotations": len(members),
                           "n_families": int(metadata.loc[members, "family"].nunique()),
                           "sse_model_total": model["sse_model_total"],
                           "sse_mean_total": model["sse_mean_total"],
                           "apparent_sse_improvement": 1 - model["sse_model_total"] / model["sse_mean_total"],
                           "model": model})
    ranked = sorted(candidates, key=lambda row: (row["sse_model_total"], tuple(sorted(row["members"]))))
    selected = ranked[0]["state_id"] if ranked else None
    score = axes["train_scores"][selected] if selected else pd.Series(0.0, index=x.index)
    return {"axes": axes, "selected_state_id": selected, "candidates": candidates,
            "model": fit_ols(score, y), "no_state_fallback": selected is None}


def parameter_record(fitted, genus, scheme, fold_id, test_ids):
    a = fitted["axes"]
    return {"genus": genus, "scheme": scheme, "fold_id": fold_id,
            "train_ids": a["train_ids"], "test_ids": list(test_ids),
            "selected_state_id": fitted["selected_state_id"],
            "selected_members": a["module_members"].get(fitted["selected_state_id"], []),
            "no_state_fallback": fitted["no_state_fallback"], "selection": SELECTION,
            "neuron_order": NEURONS, "neural_coordinate_scaling": "none",
            "source_features": a["source_features"], "retained_features": a["retained_features"],
            "excluded_features": a["excluded_features"], "feature_order": a["feature_order"],
            "training_log2_means": a["means"].to_dict(),
            "training_log2_sample_sds": a["scales"].to_dict(),
            "module_members": a["module_members"],
            "score_weights": {k: v.to_dict() for k, v in a["score_weights"].items()},
            "chemical_parameters": a["parameters"],
            "training_all_state_scores": a["train_scores"].to_dict(orient="index"),
            "candidates": fitted["candidates"], "model": fitted["model"]}


def pooled_performance(errors):
    rows = []
    for (genus, scheme), block in errors.groupby(["genus", "scheme"], sort=False):
        for neuron, subset in [("all13", block)] + list(block.groupby("neuron", sort=False)):
            sse, mean_sse = subset[["squared_error_model", "squared_error_mean"]].sum()
            n_strains = subset.strain.nunique()
            rows.append({"genus": genus, "scheme": scheme, "neuron": neuron,
                         "n_strains": n_strains, "n_coordinates": 13 if neuron == "all13" else 1,
                         "n_values": len(subset), "sse_model": float(sse),
                         "sse_training_mean": float(mean_sse),
                         "error_improvement": float(1 - sse / mean_sse) if mean_sse > 0 else None,
                         "rmse_per_coordinate_model": float(np.sqrt(sse / len(subset))),
                         "rmse_per_coordinate_mean": float(np.sqrt(mean_sse / len(subset))),
                         "rmse_vector_model": float(np.sqrt(sse / n_strains)),
                         "rmse_vector_mean": float(np.sqrt(mean_sse / n_strains))})
    return pd.DataFrame(rows)


def run_analysis(repo_root, out=None, *, overwrite=False):
    """Run the predeclared two-genus analysis and return its JSON summary.

    Chemical input is log2(c / (1 ng/mL)), 106 x 162. Neural input is the
    existing 106 x 13 unit-normalized signed template coefficient matrix.
    The two eligible genera supply 40 rows. No neural coordinates are scaled.
    """
    repo = Path(repo_root).resolve()
    output = Path(out).resolve() if out else repo / "reports/poster_local_chemical_neural_20261003"
    if not overwrite and ((output / "analysis_summary.json").exists() or
                          (output / "tables/strain_scores.csv").exists() or
                          (output / "tables/input_chemical_log2_40x162.csv").exists()):
        raise FileExistsError(f"Existing analysis outputs at {output}; pass overwrite=True explicitly to rerun")
    protocol = output / "PROTOCOL.md"
    if not protocol.is_file():
        raise FileNotFoundError("Write the bounded analysis protocol before execution")
    data_dir = repo / "reports/exploration_chemical_pattern_direct_report_20261003/tables"
    source_names = ["fresh_chemical_log2.csv", "fresh_feature_metadata.csv", "neural_unit_coefficients.csv",
                    "neural_pre_gate_unit_coefficients.csv", "sample_context.csv"]
    sources = {name: data_dir / name for name in source_names}
    sources["gated_strain_coefficients.csv"] = repo / "reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv"
    axes_api, axes_source = load_axes_api(repo)
    sources.update({"protocol": protocol, "reused_chemical_source": axes_source,
                    "analysis_source": Path(__file__).resolve()})
    hashes_before = {name: sha256(path) for name, path in sources.items()}
    x0 = pd.read_csv(sources["fresh_chemical_log2.csv"], index_col="strain")
    metadata = pd.read_csv(sources["fresh_feature_metadata.csv"], index_col="metabolite")
    y0 = pd.read_csv(sources["neural_unit_coefficients.csv"], index_col="strain")
    pre0 = pd.read_csv(sources["neural_pre_gate_unit_coefficients.csv"], index_col="strain")
    raw0 = pd.read_csv(sources["gated_strain_coefficients.csv"], index_col="strain")
    context0 = pd.read_csv(sources["sample_context.csv"], index_col="strain", dtype={"dates": str})
    assert x0.shape == (106, 162) and y0.shape == pre0.shape == (106, 13)
    assert list(y0.columns) == list(pre0.columns) == NEURONS
    assert all(frame.index.is_unique for frame in [x0, metadata, y0, pre0, context0])
    assert set(x0.index) == set(y0.index) == set(pre0.index) == set(context0.index)
    assert set(x0.columns) == set(metadata.index)
    for frame in [x0, y0, pre0]:
        assert np.isfinite(frame.to_numpy()).all()
    np.testing.assert_allclose(np.linalg.norm(y0, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(pre0, axis=1), 1.0, atol=1e-12)
    coverage = context0.groupby("genus").size().sort_values(ascending=False).rename("n_strains").reset_index()
    genera = sorted(coverage.loc[coverage.n_strains >= 10, "genus"])
    assert genera == ["Bacteroides", "Bifidobacterium"]
    metadata = metadata.loc[x0.columns]
    ids_all = sorted(context0.index[context0.genus.isin(genera)])
    assert raw0.index.is_unique and list(raw0.columns) == NEURONS
    raw = raw0.loc[ids_all]
    assert np.isfinite(raw.to_numpy()).all()
    raw_norm = np.linalg.norm(raw, axis=1)
    assert (raw_norm > 0).all()
    normalized_raw = raw.to_numpy() / raw_norm[:, None]
    raw_identity_error = float(np.max(np.abs(normalized_raw - y0.loc[ids_all].to_numpy())))
    np.testing.assert_allclose(normalized_raw, y0.loc[ids_all], rtol=1e-10, atol=1e-10)
    context = context0.loc[ids_all].copy()
    context["taxonomy_flag"] = context.taxonomy_note.fillna("").astype(str).str.strip().ne("")
    assert context.species.notna().all()
    for folder in ["tables", "parameters"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    # These focused slices plus each fold's exact parameters permit rebuilding.
    x0.loc[ids_all].to_csv(tables / "input_chemical_log2_40x162.csv")
    y0.loc[ids_all].to_csv(tables / "input_neural_unit_40x13.csv")
    pre0.loc[ids_all].to_csv(tables / "input_neural_pre_gate_unit_40x13.csv")
    raw.to_csv(tables / "input_neural_gated_coefficients_40x13.csv")
    metadata.to_csv(tables / "input_feature_metadata_162.csv")
    context.to_csv(tables / "input_context_40.csv")
    coverage["eligible"] = coverage.genus.isin(genera)
    coverage.to_csv(tables / "genus_coverage.csv", index=False)
    stores = {name: [] for name in ["strain_scores", "neural_slopes", "selected_members", "group_neural_means",
              "group_chemical_means", "selected_chemical_z", "full_candidates", "all_candidate_scores",
              "heldout_predictions", "fold_selections", "fixed_axis_deletion", "pre_gate_fixed_score_slopes",
              "gated_raw_fixed_score_slopes"]}
    summaries = {}
    for genus in genera:
        ids = sorted(context.index[context.genus.eq(genus)])
        x, y, pre, ctx = x0.loc[ids], y0.loc[ids], pre0.loc[ids], context.loc[ids]
        full = fit_selected_state(x, metadata, y, axes_api)
        axes, state, model = full["axes"], full["selected_state_id"], full["model"]
        if state is None:
            raise RuntimeError(f"No full-data eligible chemical state for {genus}")
        members = axes["module_members"][state]
        score = axes["train_scores"][state]
        iqr = float(score.quantile(0.75) - score.quantile(0.25))
        slope = np.array(model["slope"])
        norm = float(np.linalg.norm(slope))
        direction = slope / norm if norm > 0 else np.zeros(13)
        score_frame = ctx.copy()
        score_frame["state_id"] = state
        score_frame["chemical_score"] = score
        score_frame["fitted_direction_projection"] = (y - y.mean()).to_numpy() @ direction
        score_frame["unit_ADF_minus_ASH"] = y.ADF - y.ASH
        for neuron in NEURONS:
            score_frame[f"unit_{neuron}"] = y[neuron]
        order = sorted(ids, key=lambda strain: (float(score.loc[strain]), strain))
        for rank_group, group_ids in zip(["Low", "Mid", "High"], np.array_split(np.array(order), 3)):
            score_frame.loc[group_ids, "rank_group"] = rank_group
        score_frame["chemical_rank"] = [order.index(strain) + 1 for strain in ids]
        stores["strain_scores"].extend(score_frame.reset_index().to_dict("records"))
        for j, neuron in enumerate(NEURONS):
            stores["neural_slopes"].append({"genus": genus, "neuron": neuron,
                "intercept": model["intercept"][j], "slope": model["slope"][j],
                "genus_mean": model["training_mean"][j], "score_iqr": iqr,
                "iqr_effect": model["slope"][j] * iqr, "pearson_r": model["pearson_r"][j],
                "sse_model": model["sse_model"][j], "sse_mean": model["sse_mean"][j],
                "fitted_direction_loading": float(direction[j])})
        for metabolite in members:
            stores["selected_members"].append({"genus": genus, "state_id": state, "metabolite": metabolite,
                "weight": float(axes["score_weights"][state].loc[metabolite]),
                "training_log2_mean": float(axes["means"].loc[metabolite]),
                "training_log2_sample_sd": float(axes["scales"].loc[metabolite]),
                **metadata.loc[metabolite].to_dict()})
        z = (x.loc[:, members] - axes["means"].loc[members]) / axes["scales"].loc[members]
        for rank_group in ["Low", "Mid", "High"]:
            gids = score_frame.index[score_frame.rank_group.eq(rank_group)]
            for neuron in NEURONS:
                observed_mean = float(y.loc[gids, neuron].mean())
                stores["group_neural_means"].append({"genus": genus, "rank_group": rank_group,
                    "neuron": neuron, "n": len(gids), "n_species": int(ctx.loc[gids, "species"].nunique()),
                    "observed_mean": observed_mean, "genus_mean": float(y[neuron].mean()),
                    "centered_mean": observed_mean - float(y[neuron].mean())})
            for metabolite in members:
                stores["group_chemical_means"].append({"genus": genus, "rank_group": rank_group,
                    "metabolite": metabolite, "n": len(gids), "mean_z": float(z.loc[gids, metabolite].mean())})
                for strain in gids:
                    stores["selected_chemical_z"].append({"genus": genus, "strain": strain,
                        "rank_group": rank_group, "metabolite": metabolite, "z": float(z.loc[strain, metabolite])})
        for candidate in full["candidates"]:
            stores["full_candidates"].append({"genus": genus,
                **{k: v for k, v in candidate.items() if k not in ["members", "model"]},
                "members_json": json.dumps(candidate["members"]), "selected": candidate["state_id"] == state})
        for strain in ids:
            for candidate, value in axes["train_scores"].loc[strain].items():
                stores["all_candidate_scores"].append({"genus": genus, "strain": strain,
                                                      "state_id": candidate, "chemical_score": float(value)})
        full_params = parameter_record(full, genus, "full_data_descriptive", "full", [])
        full_params.update({"fitted_direction": direction.tolist(),
                            "fitted_direction_disclosure": "Full-data selected fitted direction; not independent validation"})
        write_json(output / f"parameters/{genus}_full.json", full_params)
        fold_params = []
        for scheme in SCHEMES:
            labels = pd.Series(ids, index=ids) if scheme == "leave_one_strain_out" else ctx.species
            for k, omitted in enumerate(sorted(labels.unique()), 1):
                test_ids = labels.index[labels.eq(omitted)].tolist()
                train_ids = labels.index[~labels.eq(omitted)].tolist()
                fold_id = f"{scheme}:{k:02d}"
                fitted = fit_selected_state(x.loc[train_ids], metadata, y.loc[train_ids], axes_api)
                transformed = axes_api.transform_axes(x.loc[test_ids], fitted["axes"])
                fold_state = fitted["selected_state_id"]
                test_score = transformed[fold_state] if fold_state else pd.Series(0.0, index=test_ids)
                fold_model = fitted["model"]
                predicted = np.array(fold_model["intercept"]) + test_score.to_numpy()[:, None] * np.array(fold_model["slope"])
                fold_members = fitted["axes"]["module_members"].get(fold_state, [])
                intersection = set(members) & set(fold_members)
                union = set(members) | set(fold_members)
                stores["fold_selections"].append({"genus": genus, "scheme": scheme, "fold_id": fold_id,
                    "omitted_label": omitted, "n_train": len(train_ids), "n_test": len(test_ids),
                    "train_ids_json": json.dumps(train_ids), "test_ids_json": json.dumps(test_ids),
                    "selected_state_id": fold_state, "members_json": json.dumps(fold_members),
                    "n_candidates": len(fitted["candidates"]), "no_state_fallback": fitted["no_state_fallback"],
                    "full_member_intersection": len(intersection), "full_member_union": len(union),
                    "full_member_jaccard": len(intersection) / len(union),
                    "full_members_exact_match": set(members) == set(fold_members)})
                record = parameter_record(fitted, genus, scheme, fold_id, test_ids)
                record["heldout_selected_scores"] = test_score.to_dict()
                fold_params.append(record)
                for i, strain in enumerate(test_ids):
                    for j, neuron in enumerate(NEURONS):
                        observed, prediction, baseline = float(y.loc[strain, neuron]), float(predicted[i, j]), float(fold_model["training_mean"][j])
                        stores["heldout_predictions"].append({"genus": genus, "scheme": scheme,
                            "fold_id": fold_id, "strain": strain, "species": ctx.loc[strain, "species"],
                            "neuron": neuron, "chemical_score": float(test_score.loc[strain]),
                            "observed": observed, "model_prediction": prediction, "training_mean_prediction": baseline,
                            "squared_error_model": (observed - prediction) ** 2,
                            "squared_error_mean": (observed - baseline) ** 2})
                # Descriptive influence of deleting rows with the full-data axis held fixed.
                deleted = fit_ols(score.loc[train_ids], y.loc[train_ids])
                dslope = np.array(deleted["slope"])
                dnorm = np.linalg.norm(dslope)
                cosine = float(dslope @ direction / dnorm) if dnorm > 0 else None
                for j, neuron in enumerate(NEURONS):
                    stores["fixed_axis_deletion"].append({"genus": genus, "scheme": scheme,
                        "fold_id": fold_id, "omitted_label": omitted, "neuron": neuron,
                        "slope": deleted["slope"][j], "iqr_effect_full_score_iqr": deleted["slope"][j] * iqr,
                        "pearson_r": deleted["pearson_r"][j], "slope_direction_cosine_to_full": cosine})
        write_json(output / f"parameters/{genus}_folds.json", fold_params)
        pre_model = fit_ols(score, pre)
        pre_slope = np.array(pre_model["slope"])
        pre_norm = np.linalg.norm(pre_slope)
        pre_cosine = float(pre_slope @ direction / pre_norm) if pre_norm > 0 else None
        for j, neuron in enumerate(NEURONS):
            stores["pre_gate_fixed_score_slopes"].append({"genus": genus, "neuron": neuron,
                "intercept": pre_model["intercept"][j], "slope": pre_model["slope"][j],
                "iqr_effect": pre_model["slope"][j] * iqr, "pearson_r": pre_model["pearson_r"][j],
                "direction_cosine_to_gated": pre_cosine})
        raw_model = fit_ols(score, raw.loc[ids])
        raw_slope = np.array(raw_model["slope"])
        raw_slope_norm = np.linalg.norm(raw_slope)
        raw_cosine = float(raw_slope @ direction / raw_slope_norm) if raw_slope_norm > 0 else None
        for j, neuron in enumerate(NEURONS):
            stores["gated_raw_fixed_score_slopes"].append({"genus": genus, "neuron": neuron,
                "intercept": raw_model["intercept"][j], "slope": raw_model["slope"][j],
                "iqr_effect": raw_model["slope"][j] * iqr, "pearson_r": raw_model["pearson_r"][j],
                "direction_cosine_to_unit": raw_cosine,
                "representation": "signed SNR-gated template coefficient before strain-unit normalization"})
        summaries[genus] = {"n_strains": len(ids), "n_species": int(ctx.species.nunique()),
            "n_taxonomy_flags": int(ctx.taxonomy_flag.sum()), "selected_state_id": state,
            "n_candidates": len(full["candidates"]), "n_annotations": len(members),
            "n_families": int(metadata.loc[members, "family"].nunique()), "selected_members": members,
            "score_iqr": iqr, "apparent_all13_sse_improvement": 1 - model["sse_model_total"] / model["sse_mean_total"],
            "group_counts": score_frame.rank_group.value_counts().to_dict(),
            "pre_gate_fixed_score_direction_cosine": pre_cosine,
            "gated_raw_fixed_score_direction_cosine": raw_cosine}
    for name, records in stores.items():
        pd.DataFrame(records).to_csv(tables / f"{name}.csv", index=False)
    errors = pd.DataFrame(stores["heldout_predictions"])
    performance = pooled_performance(errors)
    performance.to_csv(tables / "heldout_performance.csv", index=False)
    folds = pd.DataFrame(stores["fold_selections"])
    for genus in genera:
        summaries[genus]["heldout_all13"] = performance[(performance.genus == genus) & (performance.neuron == "all13")].to_dict("records")
        summaries[genus]["membership_stability"] = {scheme: {
            "n_folds": int(len(block)), "n_exact_match": int(block.full_members_exact_match.sum()),
            "median_jaccard": float(block.full_member_jaccard.median()),
            "min_jaccard": float(block.full_member_jaccard.min()),
            "n_mean_fallback": int(block.no_state_fallback.sum())}
            for scheme, block in folds[folds.genus.eq(genus)].groupby("scheme")}
    hashes_after = {name: sha256(path) for name, path in sources.items()}
    assert hashes_before == hashes_after, "A source changed during the run"
    manifest = {"sources": {name: {"path": str(path), "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name]} for name, path in sources.items()},
                "sources_unchanged": True, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    write_json(output / "source_manifest.json", manifest)
    summary = {"selection": SELECTION, "genus_eligibility": "at least 10 strains, based only on coverage",
               "neuron_order": NEURONS, "n_total_strains": len(ids_all), "genera": summaries,
               "limitations": ["Full-data selection, fitted directions and rank-group differences are descriptive",
                  "Leaveout refits all chemical operations and selection, conditional on saved neural representation and global chemical QC",
                  "Positive total-vector error improvement does not imply information in every coordinate",
                  "No independent experimental validation or mechanistic attribution"],
               "sources_unchanged": True,
               "gated_raw_normalization_max_abs_difference_from_unit": raw_identity_error}
    write_json(output / "analysis_summary.json", summary)
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    result = run_analysis(root)
    print(json.dumps({genus: {"state": values["selected_state_id"],
                    "annotations": values["n_annotations"], "heldout_all13": values["heldout_all13"]}
                    for genus, values in result["genera"].items()}, indent=2))

"""Descriptive within/between-genus comparisons, callable from a Notebook.

Inputs are saved strain-level data; no templates, scores or models are fitted.
Scientific outputs are written only to a new tables/results destination.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs(root):
    root = Path(root)
    source = root / "reports/exploration_chemical_pattern_direct_report_20261003/tables"
    neural_source = root / "reports/exploration_response_profiles_individual_snr_20261002/tables"
    paths = {
        "chemical": source / "fresh_chemical_log2.csv",
        "neural": source / "neural_unit_coefficients.csv",
        "pre_gate": source / "neural_pre_gate_unit_coefficients.csv",
        "context": source / "sample_context.csv",
        "features": source / "fresh_feature_metadata.csv",
        "coefficients": neural_source / "strain_coefficients.csv",
        "neural_audit": neural_source / "strain_audit.csv",
    }
    frames = {k: pd.read_csv(paths[k], index_col="strain")
              for k in ("chemical", "neural", "pre_gate", "context", "coefficients")}
    ids = sorted(frames["context"].index)
    for key, frame in frames.items():
        if not frame.index.is_unique or set(frame.index) != set(ids):
            raise ValueError(f"Strain alignment failed: {key}")
        frames[key] = frame.loc[ids]
        if key != "context" and not np.isfinite(frames[key].to_numpy()).all():
            raise ValueError(f"Nonfinite input: {key}")
    if len(ids) != 106 or frames["chemical"].shape[1] != 162 or frames["neural"].shape[1] != 13:
        raise ValueError("Expected 106 strains, 162 chemicals and 13 neural coordinates.")
    if frames["context"].genus.isna().any():
        raise ValueError("Missing genus requires an explicit handling decision.")
    for key in ("neural", "pre_gate"):
        if list(frames[key].columns) != list(frames["coefficients"].columns):
            raise ValueError("Neural coordinate order mismatch.")
        if not np.allclose(np.linalg.norm(frames[key], axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError(f"Non-unit neural vectors: {key}")
    coefficients = frames["coefficients"].to_numpy()
    normalized = coefficients / np.linalg.norm(coefficients, axis=1, keepdims=True)
    if not np.allclose(normalized, frames["neural"], atol=1e-12, rtol=0):
        raise ValueError("Saved unit vectors do not reproduce from source coefficients.")
    frames["features"] = pd.read_csv(paths["features"])
    if frames["features"].metabolite.tolist() != frames["chemical"].columns.tolist():
        raise ValueError("Chemical metadata order mismatch.")
    frames["neural_audit"] = pd.read_csv(paths["neural_audit"])
    return frames, paths


def pair_catalogue(frames):
    chemistry = frames["chemical"].to_numpy()
    matrices = {
        "chemical": squareform(pdist(chemistry, metric="euclidean")) / np.sqrt(chemistry.shape[1]),
        "neural": np.clip(squareform(pdist(frames["neural"], metric="cosine")), 0, 2),
        "neural_pre_gate": np.clip(squareform(pdist(frames["pre_gate"], metric="cosine")), 0, 2),
    }
    ids = frames["context"].index.to_numpy()
    genera = frames["context"].genus.to_numpy()
    i, j = np.triu_indices(len(ids), k=1)
    pairs = pd.DataFrame({"strain_a": ids[i], "strain_b": ids[j],
                          "genus_a": genera[i], "genus_b": genera[j],
                          "same_genus": genera[i] == genera[j]})
    for name, values in matrices.items():
        pairs[name + "_distance"] = values[i, j]
    return pairs, matrices


def comparison_pairs(pairs, genus, allowed_external=None):
    within = pairs.loc[(pairs.genus_a == genus) & (pairs.genus_b == genus)].copy()
    between = pairs.loc[(pairs.genus_a == genus) ^ (pairs.genus_b == genus)].copy()
    between["external_genus"] = np.where(between.genus_a == genus, between.genus_b, between.genus_a)
    if allowed_external is not None:
        between = between.loc[between.external_genus.isin(allowed_external)].copy()
    external_counts = between.external_genus.value_counts()
    between["weight"] = 1 / (len(external_counts) * between.external_genus.map(external_counts))
    within["weight"] = 1 / len(within)
    return within, between


def weighted_quantiles(values, weights, probabilities=(.1, .25, .5, .75, .9)):
    """Empirical quantiles; median is midpoint if exactly half the mass is below.

    The median convention reproduces np.median for equal weights, including
    even sample sizes. Other quantiles use the inverse empirical CDF.
    """
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights / weights.sum())
    indices = np.searchsorted(cumulative + 1e-12, probabilities, side="left")
    indices = np.minimum(indices, len(values) - 1)
    result = values[indices].copy()
    for position, probability in enumerate(probabilities):
        index = indices[position]
        if (probability == .5 and index < len(values) - 1
                and abs(cumulative[index] - .5) < 1e-12):
            result[position] = (values[index] + values[index + 1]) / 2
    return result


def distribution_row(group, modality):
    values, weights = group[modality + "_distance"].to_numpy(), group.weight.to_numpy()
    q = weighted_quantiles(values, weights)
    return {"n_pairs": len(group), "q10": q[0], "q25": q[1], "q50": q[2],
            "q75": q[3], "q90": q[4], "minimum": values.min(), "maximum": values.max(),
            "mean": np.average(values, weights=weights)}


def probability_within_smaller(within, between, modality):
    a = within[modality + "_distance"].to_numpy()[:, None]
    b = between[modality + "_distance"].to_numpy()[None, :]
    comparisons = (a < b).astype(float) + .5 * (a == b)
    return float(np.average(comparisons.mean(axis=0), weights=between.weight))


def summarize(pairs, coverage, allowed_external=None, modalities=("chemical", "neural")):
    rows, summaries, memberships = [], [], []
    for record in coverage.loc[coverage.n_strains >= 2].itertuples(index=False):
        genus, n = record.genus, record.n_strains
        within, between = comparison_pairs(pairs, genus, allowed_external)
        summary = {"genus": genus, "n_strains": int(n), "n_within_pairs": len(within),
                   "n_between_pairs": len(between), "n_external_genera": between.external_genus.nunique()}
        for modality in modalities:
            estimates = {}
            for relation, frame in (("within", within), ("between", between)):
                d = distribution_row(frame, modality)
                rows.append({"genus": genus, "n_strains": int(n), "modality": modality,
                             "relation": relation, **d})
                estimates[relation] = d
                summary[f"{modality}_{relation}_median"] = d["q50"]
            summary[f"{modality}_median_ratio"] = estimates["within"]["q50"] / estimates["between"]["q50"]
            summary[f"{modality}_prob_within_smaller"] = probability_within_smaller(within, between, modality)
        summaries.append(summary)
        for relation, frame in (("within", within), ("between", between)):
            members = frame[["strain_a", "strain_b", "weight"]].copy()
            members.insert(0, "relation", relation)
            members.insert(0, "focal_genus", genus)
            memberships.append(members)
    return pd.DataFrame(rows), pd.DataFrame(summaries), pd.concat(memberships, ignore_index=True)


def sample_qc(frames):
    qc = frames["context"].copy()
    qc["coefficient_l2_norm"] = np.linalg.norm(frames["coefficients"], axis=1)
    qc["nonzero_neural_coordinates"] = (frames["coefficients"] != 0).sum(axis=1)
    qc["n_dates"] = qc.dates.astype(str).str.split(";").str.len()
    audit = frames["neural_audit"].groupby("strain")
    qc["min_animals_per_cell_total"] = audit.n_animals_recorded.min()
    qc["max_animals_per_cell_total"] = audit.n_animals_recorded.max()
    return qc


def run_analysis(root, out):
    """Compute only descriptive steps 1/2 and two specified sensitivity checks.

    Chemistry: RMS differences in log2(concentration / (1 ng/mL)).
    Neural: 1-cosine of 13 unit coefficients (overall gain removed).
    External genera receive equal mass; pairs are not independent replicates.
    """
    root, out = Path(root), Path(out)
    if (out / "tables").exists() or (out / "results.json").exists():
        raise FileExistsError("Use a fresh output directory; existing scientific outputs are protected.")
    frames, paths = load_inputs(root)
    hashes = {str(path): file_hash(path) for path in paths.values()}
    coverage = (frames["context"].genus.value_counts().rename_axis("genus")
                .reset_index(name="n_strains").sort_values(["n_strains", "genus"], ascending=[False, True]))
    coverage["n_within_pairs"] = coverage.n_strains * (coverage.n_strains - 1) // 2
    coverage["role"] = np.where(coverage.n_strains >= 2, "within_and_external", "external_only")
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "scope": "Descriptive steps 1 and 2 only; no prediction, clustering, hypothesis tests or steps 3-5.",
        "n_strains": 106, "n_chemical_annotations": 162, "n_neural_coordinates": 13,
        "chemical_distance": "sqrt(mean_k((log2(c_a[k]/(1 ng/mL))-log2(c_b[k]/(1 ng/mL)))**2))",
        "chemical_weights": "Equal per annotation; no variance scaling, family reweighting, pseudocount or reference denominator.",
        "neural_distance": "1-cosine; saved SNR-gated unit coefficients; no template refit or noise subtraction.",
        "within": "All unordered distinct strain pairs within each genus with n>=2; equal pair weight.",
        "between": "For each focal genus, all pairs to 28 other genera. External genera equal weight; pairs equal within each external genus.",
        "singletons": "16 singleton genera included only as external references; no within-genus estimate.",
        "display_order": "Decreasing strain count, then alphabetical genus name; same order in both panels.",
        "quantiles": "Inverse weighted empirical CDF; at exactly 0.5 cumulative mass the median is the midpoint of adjacent values (usual median for equal weights); 1e-12 CDF tolerance.",
        "intervals": "IQR and 10th-90th distance percentiles, not confidence intervals.",
        "probability": "P(random within pair distance < genus-balanced between pair distance) + half ties; descriptive only.",
        "sensitivities": ["Use only the other 12 multi-strain genera as external references.",
                          "Use saved pre-gate coefficients on the SAME neural templates, unit normalized; all 28 external genera."],
        "inference": "No pair-independent tests or confidence intervals. Any across-genus summaries weight genera equally.",
        "context": "No same-date or old-reference filter; dates retained as metadata. No experimental confound adjustment.",
        "source_sha256": hashes,
    }
    save_json(out / "protocol.json", protocol)
    tables = out / "tables"
    tables.mkdir()
    pairs, matrices = pair_catalogue(frames)
    distributions, summary, membership = summarize(pairs, coverage)
    repeated = coverage.loc[coverage.n_strains >= 2, "genus"].tolist()
    _, repeated_summary, _ = summarize(pairs, coverage, allowed_external=repeated)
    _, pre_gate_summary, _ = summarize(pairs, coverage, modalities=("neural_pre_gate",))
    repeated_summary.insert(0, "sensitivity", "external_multistrain_genera_only")
    pre_gate_summary.insert(0, "sensitivity", "neural_before_gate_same_templates")
    for name, frame in (("pair_catalogue", pairs), ("genus_coverage", coverage),
                        ("distribution_summary", distributions), ("genus_summary", summary),
                        ("comparison_membership_weights", membership),
                        ("sensitivity_external_multistrain", repeated_summary),
                        ("sensitivity_neural_pre_gate", pre_gate_summary)):
        frame.to_csv(tables / f"{name}.csv", index=False)
    qc = sample_qc(frames)
    qc.to_csv(tables / "sample_context_qc.csv")
    frames["features"].to_csv(tables / "chemical_feature_metadata.csv", index=False)
    for modality, matrix in matrices.items():
        pd.DataFrame(matrix, index=frames["context"].index, columns=frames["context"].index).to_csv(
            tables / f"{modality}_distance_matrix.csv")
    findings = {"n_strains": 106, "n_genera": len(coverage), "n_multi_strain_genera": len(summary),
                "n_singleton_genera": int((coverage.n_strains == 1).sum()),
                "n_all_pairs": len(pairs), "n_within_pairs": int(pairs.same_genus.sum()),
                "descriptive_only": True, "by_modality": {}}
    for modality in ("chemical", "neural"):
        ratios = summary[f"{modality}_median_ratio"]
        findings["by_modality"][modality] = {
            "genera_with_within_median_below_between": int((ratios < 1).sum()),
            "median_of_genus_median_ratios": float(ratios.median()),
            "genus_median_ratio_range": [float(ratios.min()), float(ratios.max())],
            "median_of_genus_probabilities": float(summary[f"{modality}_prob_within_smaller"].median()),
        }
    findings["sensitivities"] = {
        "external_multistrain_genera_only": {
            modality: {"genera_with_within_median_below_between": int((repeated_summary[f"{modality}_median_ratio"] < 1).sum()),
                       "median_of_genus_median_ratios": float(repeated_summary[f"{modality}_median_ratio"].median())}
            for modality in ("chemical", "neural")},
        "neural_before_gate_same_templates": {
            "genera_with_within_median_below_between": int((pre_gate_summary.neural_pre_gate_median_ratio < 1).sum()),
            "median_of_genus_median_ratios": float(pre_gate_summary.neural_pre_gate_median_ratio.median())},
    }
    if any(file_hash(path) != hashes[str(path)] for path in paths.values()):
        raise RuntimeError("An input changed during analysis.")
    save_json(out / "results.json", findings)
    return findings

"""Independent small-array checks against saved inputs; no model fitting."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def check_results(root, out):
    root, out = Path(root), Path(out)
    source = root / "reports/exploration_chemical_pattern_direct_report_20261003/tables"
    chemical = pd.read_csv(source / "fresh_chemical_log2.csv", index_col="strain").sort_index()
    neural = pd.read_csv(source / "neural_unit_coefficients.csv", index_col="strain").loc[chemical.index]
    context = pd.read_csv(source / "sample_context.csv", index_col="strain").loc[chemical.index]
    x, u = chemical.to_numpy(), neural.to_numpy()
    direct = {
        "chemical": np.sqrt(np.mean((x[:, None, :] - x[None, :, :]) ** 2, axis=2)),
        "neural": np.clip(1 - (u @ u.T) / np.outer(np.linalg.norm(u, axis=1), np.linalg.norm(u, axis=1)), 0, 2),
    }
    pairs = pd.read_csv(out / "tables/pair_catalogue.csv")
    summary = pd.read_csv(out / "tables/genus_summary.csv").set_index("genus")
    distribution = pd.read_csv(out / "tables/distribution_summary.csv")
    membership = pd.read_csv(out / "tables/comparison_membership_weights.csv")
    positions = {strain: i for i, strain in enumerate(chemical.index)}
    a = pairs.strain_a.map(positions).to_numpy()
    b = pairs.strain_b.map(positions).to_numpy()
    errors = {m: float(np.max(np.abs(matrix[a, b] - pairs[m + "_distance"]))) for m, matrix in direct.items()}
    assert len(pairs) == 5565 and int(pairs.same_genus.sum()) == 561
    assert not pairs[["strain_a", "strain_b"]].duplicated().any()
    assert np.all(a < b)
    assert max(errors.values()) < 1e-12
    median_errors, probability_errors, weight_errors = [], [], []
    genera = context.genus.to_numpy()
    for genus, row in summary.iterrows():
        inside = np.flatnonzero(genera == genus)
        others = sorted(set(genera) - {genus})
        assert len(inside) == row.n_strains
        assert row.n_within_pairs == len(inside) * (len(inside) - 1) // 2
        assert row.n_between_pairs == len(inside) * (len(genera) - len(inside))
        for modality, matrix in direct.items():
            within_matrix = matrix[np.ix_(inside, inside)]
            within = within_matrix[np.triu_indices(len(inside), k=1)]
            med = float(np.median(within))
            median_errors.append(abs(med - row[f"{modality}_within_median"]))
            shown = distribution.loc[(distribution.genus == genus) & (distribution.modality == modality)
                                     & (distribution.relation == "within"), "q50"].item()
            median_errors.append(abs(shown - med))
            probabilities = []
            external_values, external_weights = [], []
            for other in others:
                outside = np.flatnonzero(genera == other)
                values = matrix[np.ix_(inside, outside)].ravel()
                comparisons = (within[:, None] < values[None, :]).astype(float)
                comparisons += .5 * (within[:, None] == values[None, :])
                probabilities.append(comparisons.mean())
                external_values.extend(values)
                external_weights.extend(np.full(len(values), 1 / (len(others) * len(values))))
            probability_errors.append(abs(np.mean(probabilities) - row[f"{modality}_prob_within_smaller"]))
            order = np.argsort(external_values)
            sorted_values = np.asarray(external_values)[order]
            cumulative = np.cumsum(np.asarray(external_weights)[order])
            k = np.flatnonzero(cumulative >= .5 - 1e-12)[0]
            between_median = sorted_values[k]
            if abs(cumulative[k] - .5) < 1e-12:
                between_median = (sorted_values[k] + sorted_values[k + 1]) / 2
            median_errors.append(abs(between_median - row[f"{modality}_between_median"]))
            assert abs(med / between_median - row[f"{modality}_median_ratio"]) < 1e-12
        weights = membership.loc[membership.focal_genus == genus].copy()
        for relation in ("within", "between"):
            weight_errors.append(abs(weights.loc[weights.relation == relation, "weight"].sum() - 1))
        external = weights.loc[weights.relation == "between"].copy()
        left = external.strain_a.map(context.genus)
        right = external.strain_b.map(context.genus)
        external["external_genus"] = np.where(left == genus, right, left)
        weight_errors.extend(abs(external.groupby("external_genus").weight.sum() - 1 / len(others)))
    assert max(median_errors) < 1e-12 and max(probability_errors) < 1e-12
    assert max(weight_errors) < 1e-12
    protocol = json.loads((out / "protocol.json").read_text())
    source_hashes_match = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                              for p, h in protocol["source_sha256"].items())
    assert source_hashes_match
    checks = {
        "status": "passed", "n_strains": len(context), "n_all_pairs": len(pairs),
        "n_within_pairs": int(pairs.same_genus.sum()),
        "distance_max_abs_error": errors, "median_max_abs_error": float(max(median_errors)),
        "probability_max_abs_error": float(max(probability_errors)),
        "weight_max_abs_error": float(max(weight_errors)), "all_source_hashes_match": source_hashes_match,
        "method": "Distances directly from NumPy broadcasting/dot products; ordinary within medians; external genera enumerated independently; probability computed as average of 28 per-external-genus comparisons.",
        "limitations": "Numeric verification, not independent biological validation or uncertainty estimation.",
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / "code").glob("*.py"))},
    }
    (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks

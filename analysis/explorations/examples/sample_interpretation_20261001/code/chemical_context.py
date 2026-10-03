"""Small, chemistry-only context for the requested three-strain poster example.

Reuse the aligned Notebook-03 log2FC and original-report masks; do not rerun
upstream normalization. A numerical FC exists even for a missing report value.
The fixed three-strain common mask is shared by all three primary comparisons.
Selection uses chemistry alone and has no neural-response input.
"""
from pathlib import Path
import hashlib
import itertools
import json
import platform

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT / "reports/population_first_20260930/tables"
PRIMARY = ["A021", "A022", "A023"]
SUPPORT = ["A007", "A010"]
QC_MAX = 0.30
DISPLAY_N = 6


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    tables, logs = OUT / "tables", OUT / "logs"
    tables.mkdir(exist_ok=True, parents=True)
    logs.mkdir(exist_ok=True, parents=True)
    paths = {
        "fc": SOURCE / "aligned_chemical_log2fc_paired.csv",
        "observed": SOURCE / "aligned_chemical_report_observed_paired.parquet",
        "raw": SOURCE / "aligned_chemical_report_values_all.csv",
        "metadata": SOURCE / "aligned_chemical_metadata.csv",
        "reference": SOURCE / "aligned_chemical_reference_groups_all.csv",
        "taxonomy": SOURCE / "aligned_taxonomy_all.csv",
        "prior_pair_audit": ROOT / "reports/chemical_neighborhood_focus_20260930/tables/chemical_neighbor_audit_pairs.csv",
        "notebook": ROOT / "notebook/03_chemical_neuron_bacteria.ipynb",
    }
    input_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths.values()}
    fc = pd.read_csv(paths["fc"], index_col=0)
    obs = pd.read_parquet(paths["observed"])
    raw = pd.read_csv(paths["raw"], index_col=0)
    meta = pd.read_csv(paths["metadata"], index_col=0)
    ref = pd.read_csv(paths["reference"], index_col=0)
    tax = pd.read_csv(paths["taxonomy"], index_col=0)
    prior = pd.read_csv(paths["prior_pair_audit"])
    assert fc.shape == (106, 380)
    assert fc.index.equals(obs.index) and fc.columns.equals(obs.columns)
    assert fc.columns.equals(raw.columns) and fc.columns.equals(meta.index)
    assert fc.index.is_unique and fc.columns.is_unique
    assert np.isfinite(fc.to_numpy()).all()
    assert obs.dtypes.eq(bool).all()
    ids = PRIMARY + SUPPORT
    assert obs.loc[ids].equals(raw.loc[ids].notna())
    assert ref.loc[ids, "reference_group"].eq("A050").all()
    assert tax.loc[PRIMARY, "species_clean"].eq("Bacteroides stercoris").all()

    fixed_common = obs.loc[PRIMARY].all(axis=0)
    qc_ok = meta.QCRSD.le(QC_MAX) & meta.qc_n_observed.ge(2)
    profiles = meta.copy()
    for strain in PRIMARY:
        profiles[f"{strain}_log2fc"] = fc.loc[strain]
        profiles[f"{strain}_reported"] = obs.loc[strain]
        profiles[f"{strain}_report_ng_ml"] = raw.loc[strain]
    profiles["n_primary_reported"] = obs.loc[PRIMARY].sum(axis=0)
    profiles["fixed_primary_common"] = fixed_common
    profiles["qc_screen_pass"] = qc_ok
    # No ranking on the imputed features: numerical validity is not observation.
    profiles["primary_log2fc_range"] = (fc.loc[PRIMARY].max() - fc.loc[PRIMARY].min()).where(fixed_common)
    ranked = profiles.loc[fixed_common].reset_index().sort_values(
        ["primary_log2fc_range", "metabolite"], ascending=[False, True], kind="stable")
    ranked["common_range_rank"] = np.arange(1, len(ranked) + 1)
    candidates = ranked.loc[ranked.qc_screen_pass].head(DISPLAY_N).copy()
    candidates["display_rank"] = np.arange(1, len(candidates) + 1)
    profiles["candidate_display_rank"] = candidates.set_index("metabolite").display_rank
    profiles.to_csv(tables / "chemical_primary_profiles_all380.csv")
    ranked.to_csv(tables / "chemical_primary_common_ranked.csv", index=False)
    candidates.to_csv(tables / "chemical_candidate_display.csv", index=False)

    pair_rows, feature_rows, missing_rows, ecdf_rows, class_rows = [], [], [], [], []
    max_reconstruction_error, max_prior_error = 0.0, 0.0
    comparisons = [(a, b, "primary") for a, b in itertools.combinations(PRIMARY, 2)]
    comparisons.append((*SUPPORT, "support"))
    for a, b, scope in comparisons:
        pair = f"{a}_{b}"
        delta = fc.loc[a] - fc.loc[b]
        energy = delta.pow(2)
        joint = obs.loc[a] & obs.loc[b]
        only_a = obs.loc[a] & ~obs.loc[b]
        only_b = ~obs.loc[a] & obs.loc[b]
        neither = ~obs.loc[a] & ~obs.loc[b]
        # Validation only: the existing same-reference pseudocount cancels.
        reconstructed = np.log2(raw.loc[a].fillna(0) + 1) - np.log2(raw.loc[b].fillna(0) + 1)
        max_reconstruction_error = max(max_reconstruction_error, float((delta - reconstructed).abs().max()))
        common = fixed_common if scope == "primary" else joint
        common_label = "fixed_primary_common" if scope == "primary" else "support_pair_joint"
        subsets = {"all380": pd.Series(True, index=fc.columns),
                   "pair_joint": joint, common_label: common,
                   f"{common_label}_qc_screen": common & qc_ok}
        row = dict(pair_id=pair, strain_a=a, strain_b=b, scope=scope,
                   reference_group=ref.loc[a, "reference_group"],
                   n_all=len(fc.columns), n_pair_joint=int(joint.sum()),
                   n_common=int(common.sum()), common_set=common_label,
                   n_common_qc_screen=int((common & qc_ok).sum()),
                   full_rms_log2fc=rms(delta), joint_rms_log2fc=rms(delta[joint]),
                   common_rms_log2fc=rms(delta[common]),
                   common_qc_screen_rms_log2fc=rms(delta[common & qc_ok]),
                   n_only_a=int(only_a.sum()), n_only_b=int(only_b.sum()),
                   n_neither=int(neither.sum()),
                   one_missing_squared_distance_fraction=float(energy[only_a | only_b].sum() / energy.sum()),
                   common_squared_distance_fraction=float(energy[common].sum() / energy.sum()),
                   high_qcrsd_squared_distance_fraction=float(energy[meta.QCRSD.gt(QC_MAX)].sum() / energy.sum()),
                   common_fraction_abs_log2diff_le1=float(delta[common].abs().le(1).mean()),
                   common_fraction_abs_log2diff_le2=float(delta[common].abs().le(2).mean()),
                   common_median_abs_log2diff=float(delta[common].abs().median()),
                   common_q90_abs_log2diff=float(delta[common].abs().quantile(.9)),
                   common_max_abs_log2diff=float(delta[common].abs().max()),
                   common_median_fc_ratio_magnitude=float(np.exp2(delta[common].abs()).median()))
        old = prior.loc[prior.strain_a.eq(a) & prior.strain_b.eq(b)]
        if len(old):
            assert len(old) == 1
            max_prior_error = max(max_prior_error, abs(row["full_rms_log2fc"] - old.iloc[0].full_rms_log2fc))
        pair_rows.append(row)
        states = {"joint_reported": joint, "only_a_reported": only_a,
                  "only_b_reported": only_b, "neither_reported": neither}
        for label, mask in states.items():
            missing_rows.append(dict(pair_id=pair, scope=scope, state=label,
                                     n_features=int(mask.sum()),
                                     squared_distance_sum=float(energy[mask].sum()),
                                     squared_distance_fraction=float(energy[mask].sum() / energy.sum()),
                                     contribution_to_full_squared_rms=float(energy[mask].sum() / len(fc.columns))))
        for name, mask in subsets.items():
            ranked_abs = delta[mask].abs().sort_values(kind="stable")
            for k, (feature, value) in enumerate(ranked_abs.items(), 1):
                ecdf_rows.append(dict(pair_id=pair, scope=scope, feature_set=name,
                                      metabolite=feature, abs_log2fc_difference=float(value),
                                      fc_ratio_magnitude=float(np.exp2(value)),
                                      ecdf=k / len(ranked_abs), n_features=len(ranked_abs)))
        for feature in fc.columns:
            state = next(name for name, mask in states.items() if mask.loc[feature])
            feature_rows.append(dict(pair_id=pair, scope=scope, metabolite=feature,
                                     a_log2fc=fc.loc[a, feature], b_log2fc=fc.loc[b, feature],
                                     a_report_ng_ml=raw.loc[a, feature], b_report_ng_ml=raw.loc[b, feature],
                                     a_minus_b_log2fc=delta.loc[feature],
                                     abs_log2fc_difference=abs(delta.loc[feature]),
                                     fc_ratio_magnitude=np.exp2(abs(delta.loc[feature])),
                                     squared_difference=energy.loc[feature],
                                     missing_state=state, in_common_set=bool(common.loc[feature]),
                                     qc_screen_pass=bool(qc_ok.loc[feature])))
        # Class contributions retain feature multiplicity; no category activity score.
        grouped = pd.DataFrame({"Class": meta["Class"].fillna("Unannotated"),
                                "energy": energy, "included": common})
        for cls, group in grouped.loc[grouped.included].groupby("Class"):
            class_rows.append(dict(pair_id=pair, chemical_class=cls, common_set=common_label,
                                   n_features=len(group), squared_distance_sum=float(group.energy.sum()),
                                   fraction_of_common_squared_distance=float(group.energy.sum() / energy[common].sum()),
                                   within_class_rms_log2fc=float(np.sqrt(group.energy.mean()))))

    pair_table = pd.DataFrame(pair_rows)
    feature_table = pd.DataFrame(feature_rows)
    missing_table = pd.DataFrame(missing_rows)
    pair_table.to_csv(tables / "chemical_pair_summary.csv", index=False)
    feature_table.to_csv(tables / "chemical_pair_features.csv", index=False)
    missing_table.to_csv(tables / "chemical_missing_contributions.csv", index=False)
    pd.DataFrame(ecdf_rows).to_csv(tables / "chemical_pair_ecdf.csv", index=False)
    pd.DataFrame(class_rows).to_csv(tables / "chemical_class_contributions.csv", index=False)

    assert max_reconstruction_error < 1e-12 and max_prior_error < 1e-12
    for row in pair_table.itertuples():
        parts = missing_table.loc[missing_table.pair_id.eq(row.pair_id)]
        assert parts.n_features.sum() == 380
        assert np.isclose(parts.squared_distance_fraction.sum(), 1)
        assert np.isclose(parts.contribution_to_full_squared_rms.sum(), row.full_rms_log2fc ** 2)
    assert len(ranked) == int(fixed_common.sum())
    assert candidates.qc_screen_pass.all() and candidates.fixed_primary_common.all()
    assert (candidates.primary_log2fc_range.diff().dropna() <= 0).all()
    assert len(candidates) == DISPLAY_N
    inputs_unchanged = all(sha256(p) == input_hashes[str(p.relative_to(ROOT))] for p in paths.values())
    assert inputs_unchanged
    parameters = dict(
        primary_strains=PRIMARY, support_pair=SUPPORT, primary_fixed_common_n=int(fixed_common.sum()),
        observed_counts={s: int(obs.loc[s].sum()) for s in ids},
        distance="sqrt(mean((log2FC_a-log2FC_b)^2)); equal weight per report feature",
        transform="Reuse existing log2FC; no new pseudocount, imputation, scaling, or normalization",
        missing="Original report non-missing mask; upstream numerical FC already includes fill-zero/+1 convention",
        shared_reference="A050; numerical reference identifier; chemical FC relative to medium per Notebook cell 5",
        display_selection="From fixed three-strain common-report set, rank descending max-min log2FC (name tie-break); first six with QCRSD<=0.30 and >=2 QC observations. No neural data used.",
        qc_max=QC_MAX, display_n=DISPLAY_N, qc_screened_common_n=int((fixed_common & qc_ok).sum()),
        fc_ratio_magnitude="2**abs(delta_log2FC): ratio of reported fold changes, NOT measured concentration ratio; inherited +1 affects low values",
        class_summary="Descriptive sum of squared differences and within-class RMS, preserving feature count; not pathway activity or class enrichment",
        limitations=["Independent culture batch from neural stimulus; no matched aliquot concentration",
                     "No chemical biological-replicate uncertainty supplied", "No MSI identification confidence levels supplied",
                     "QC injection RSD is analytical report metadata, not biological replication or identity validation",
                     "No molecular association tests or causal interpretation; three strains do not support 380-feature screening"],
        input_sha256=input_hashes, script_sha256=sha256(Path(__file__)),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    checks = dict(status="passed", inputs_unchanged=inputs_unchanged,
                  dimensions=[106, 380], mask_equals_original_report_nan=True,
                  shared_reference_and_primary_species_verified=True,
                  maximum_same_reference_delta_reconstruction_error=max_reconstruction_error,
                  maximum_previous_pair_rms_error=max_prior_error,
                  missing_contributions_sum_to_full_squared_rms=True,
                  candidate_rule_verified=True)
    (logs / "chemical_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (logs / "chemical_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(pair_table.to_string(index=False))
    print("\nCandidate annotations (identity confidence unavailable):")
    print(candidates[["metabolite", "primary_log2fc_range", "QCRSD", "common_range_rank"]].to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

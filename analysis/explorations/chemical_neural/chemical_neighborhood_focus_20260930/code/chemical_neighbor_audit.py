"""Read-only audit of the existing 380-feature chemistry-selected pairs.

No neural models or outcomes are used. Jointly reported distances and altered
pseudocounts are sensitivity diagnostics, not replacements for Notebook FC.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
SOURCE = OUT.parent / 'population_first_20260930' / 'tables'


def select_nearest(pairs, value):
    """Union of each strain's closest partner within original comparable set."""
    selected = set()
    for _, g in pairs.groupby(['date', 'genus', 'reference_group']):
        for strain in sorted(set(g.strain_a) | set(g.strain_b)):
            eligible = g[g.strain_a.eq(strain) | g.strain_b.eq(strain)]
            selected.add(eligible.sort_values([value, 'pair_id']).iloc[0].pair_id)
    return selected


def main():
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(SOURCE / 'neural_value_pairs.csv')
    fc = pd.read_csv(SOURCE / 'aligned_chemical_log2fc_all.csv', index_col=0)
    paired_ids = pd.read_csv(SOURCE / 'aligned_chemical_log2fc_paired.csv', index_col=0).index
    raw = pd.read_csv(SOURCE / 'aligned_chemical_report_values_all.csv', index_col=0)
    observed = pd.read_parquet(SOURCE / 'aligned_chemical_report_observed_all.parquet')
    tax = pd.read_csv(SOURCE / 'aligned_taxonomy_all.csv', index_col=0)
    ref = pd.read_csv(SOURCE / 'aligned_chemical_reference_groups_all.csv', index_col=0)
    metadata = pd.read_csv(SOURCE / 'aligned_chemical_metadata.csv', index_col=0)
    assert raw.columns.equals(fc.columns) and observed.columns.equals(fc.columns)
    rows = []
    max_reconstruction_error = 0.0
    for pair in pairs.itertuples():
        a, b = pair.strain_a, pair.strain_b
        delta = fc.loc[a] - fc.loc[b]
        joint = observed.loc[a] & observed.loc[b]
        discordant = observed.loc[a] ^ observed.loc[b]
        missing_both = ~(observed.loc[a] | observed.loc[b])
        energy = float(delta.pow(2).sum())
        reconstructed = np.log2(raw.loc[a].fillna(0) + 1) - np.log2(raw.loc[b].fillna(0) + 1)
        max_reconstruction_error = max(max_reconstruction_error, float((delta - reconstructed).abs().max()))
        row = dict(pair_id=pair.pair_id, full_rms_log2fc=float(np.sqrt(delta.pow(2).mean())),
                   n_joint_reported=int(joint.sum()), n_one_missing=int(discordant.sum()),
                   n_both_missing=int(missing_both.sum()),
                   one_missing_squared_distance_fraction=float(delta[discordant].pow(2).sum()/energy),
                   high_qcrsd_squared_distance_fraction=float(delta[metadata.QCRSD.gt(.3)].pow(2).sum()/energy),
                   joint_rms_log2fc=float(np.sqrt(delta[joint].pow(2).mean())),
                   joint_median_abs_log2fc_difference=float(delta[joint].abs().median()),
                   joint_fraction_abs_difference_le_1=float(delta[joint].abs().le(1).mean()),
                   joint_fraction_abs_difference_le_2=float(delta[joint].abs().le(2).mean()),
                   joint_profile_pearson=float(np.corrcoef(fc.loc[a, joint], fc.loc[b, joint])[0, 1]),
                   top10_squared_distance_fraction=float(delta.pow(2).nlargest(10).sum()/energy))
        # Same-reference denominator cancels in the pair difference. These
        # arbitrary-unit pseudocount variations only audit ranking sensitivity.
        for pseudocount, label in [(.1, '0p1'), (10., '10')]:
            dd = np.log2(raw.loc[a].fillna(0) + pseudocount) - np.log2(raw.loc[b].fillna(0) + pseudocount)
            row[f'pseudocount_{label}_rms'] = float(np.sqrt(dd.pow(2).mean()))
        rows.append(row)
    metrics = pairs.merge(pd.DataFrame(rows), on='pair_id', validate='one_to_one')
    assert np.allclose(metrics.full_rms_log2fc, metrics.chemical_rms_log2fc, atol=1e-12, rtol=0)
    assert max_reconstruction_error < 1e-12
    old = set(metrics.loc[metrics.nearest_either.eq(1), 'pair_id'])
    sensitivity_rows = []
    for method in ['full_rms_log2fc', 'joint_rms_log2fc', 'pseudocount_0p1_rms', 'pseudocount_10_rms']:
        selected = select_nearest(metrics, method)
        metrics[method + '_nearest_either'] = metrics.pair_id.isin(selected)
        sensitivity_rows.append(dict(method=method, n_selected=len(selected),
                                     intersection_original=len(old & selected),
                                     jaccard_original=len(old & selected)/len(old | selected)))
    metrics.to_csv(OUT / 'tables/chemical_neighbor_audit_pairs.csv', index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / 'tables/chemical_neighbor_audit_sensitivity.csv', index=False)
    ranks = []
    for a, b in [('A007', 'A010'), ('A022', 'A023')]:
        match = metrics[metrics.strain_a.eq(a) & metrics.strain_b.eq(b)].iloc[0]
        same_group = metrics[metrics.date.eq(match.date) & metrics.genus.eq(match.genus)
                             & metrics.reference_group.eq(match.reference_group)]
        for source, target in [(a, b), (b, a)]:
            for method in ['full_rms_log2fc', 'joint_rms_log2fc', 'pseudocount_0p1_rms', 'pseudocount_10_rms']:
                eligible = same_group[same_group.strain_a.eq(source) | same_group.strain_b.eq(source)]
                ids = list(eligible.sort_values([method, 'pair_id']).pair_id)
                ranks.append(dict(strain=source, partner=target, universe='original_comparable_set',
                                  method=method, rank=ids.index(match.pair_id)+1, n_candidates=len(ids)))
            for universe, ids in [('paired106', paired_ids), ('all299', fc.index)]:
                eligible_ids = [i for i in ids if i != source
                                and tax.loc[i, 'genus_clean'] == tax.loc[source, 'genus_clean']
                                and ref.loc[i, 'reference_group'] == ref.loc[source, 'reference_group']]
                distances = pd.Series({i: np.sqrt((fc.loc[i]-fc.loc[source]).pow(2).mean()) for i in eligible_ids}).sort_values()
                ranks.append(dict(strain=source, partner=target, universe=universe,
                                  method='full_rms_log2fc', rank=list(distances.index).index(target)+1,
                                  n_candidates=len(distances)))
    pd.DataFrame(ranks).to_csv(OUT / 'tables/chemical_neighbor_audit_example_ranks.csv', index=False)
    nearest = metrics[metrics.nearest_either.eq(1)]
    summary = dict(n_eligible_pairs=len(metrics), n_original_neighbors=len(nearest),
                   automatic_neighbors_group_n_2=int(nearest.group_n_strains.eq(2).sum()),
                   same_reference_delta_reconstruction_max_error=max_reconstruction_error,
                   distance_recompute_max_error=float((metrics.full_rms_log2fc-metrics.chemical_rms_log2fc).abs().max()),
                   one_missing_energy_fraction_median=float(nearest.one_missing_squared_distance_fraction.median()),
                   one_missing_energy_fraction_min=float(nearest.one_missing_squared_distance_fraction.min()),
                   one_missing_energy_fraction_max=float(nearest.one_missing_squared_distance_fraction.max()),
                   status='success; no neural models or new selection by neural outcome')
    (OUT / 'tables/chemical_neighbor_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

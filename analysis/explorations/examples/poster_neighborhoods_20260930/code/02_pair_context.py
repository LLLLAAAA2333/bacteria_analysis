"""Context for chemistry-neighborhood pair comparisons; no pair-independent tests.

Pairs were fixed using chemistry and the shared stimulus catalog. This script
records species matching, observed-versus-missing chemical contributions, and
actual stimulus sequence positions before any neural outcome is considered.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = OUT.parent / 'population_first_20260930' / 'tables'
GROUP = ['date', 'genus', 'reference_group']


def context():
    pairs = pd.read_csv(SOURCE / 'neural_value_pairs.csv', dtype={'date': str})
    fc = pd.read_csv(SOURCE / 'aligned_chemical_log2fc_paired.csv', index_col=0)
    observed = pd.read_parquet(SOURCE / 'aligned_chemical_report_observed_paired.parquet')
    metadata = pd.read_csv(SOURCE / 'aligned_chemical_metadata.csv', index_col=0)
    assert observed.index.equals(fc.index) and observed.columns.equals(fc.columns)
    assert metadata.index.equals(fc.columns)
    trials = pd.read_parquet(ROOT / 'reports/exploration_20260929/tables/trial_curves.parquet').reset_index()
    trials['date'] = trials.date.astype(str)
    slots = trials[['sample_id', 'date', 'worm_key', 'segment_index']].drop_duplicates()
    # Do not count the same delivered trial once per recorded neuron.
    assert slots.groupby(['date', 'worm_key', 'segment_index']).sample_id.nunique().eq(1).all()
    positions = slots.groupby(['date', 'worm_key', 'sample_id']).segment_index.agg(['min', 'mean', 'max', 'size'])
    labels = {key: f'G{i + 1:02d}' for i, key in enumerate(sorted(pairs.groupby(GROUP).groups))}
    rows, animal_rows = [], []
    for pair in pairs.itertuples():
        a, b = pair.strain_a, pair.strain_b
        delta = (fc.loc[a] - fc.loc[b]).to_numpy(float)
        joint = (observed.loc[a] & observed.loc[b]).to_numpy(bool)
        one = (observed.loc[a] ^ observed.loc[b]).to_numpy(bool)
        both_missing = ~(observed.loc[a] | observed.loc[b]).to_numpy(bool)
        energy = np.square(delta).sum()
        p = positions.xs(pair.date, level='date')
        pa, pb = p.xs(a, level='sample_id'), p.xs(b, level='sample_id')
        animals = pa.index.intersection(pb.index)
        diff = pa.loc[animals, 'mean'] - pb.loc[animals, 'mean']
        first_diff = pa.loc[animals, 'min'] - pb.loc[animals, 'min']
        last_diff = pa.loc[animals, 'max'] - pb.loc[animals, 'max']
        for animal in animals:
            animal_rows.append(dict(pair_id=pair.pair_id, date=pair.date, worm_key=animal,
                                    mean_position_a=pa.loc[animal, 'mean'], mean_position_b=pb.loc[animal, 'mean'],
                                    mean_position_difference_a_minus_b=diff.loc[animal],
                                    first_position_difference_a_minus_b=first_diff.loc[animal],
                                    last_position_difference_a_minus_b=last_diff.loc[animal],
                                    n_trials_a=int(pa.loc[animal, 'size']), n_trials_b=int(pb.loc[animal, 'size'])))
        nonzero = np.sign(diff.to_numpy(float)[diff.to_numpy(float) != 0])
        first_sign = np.sign(first_diff.to_numpy(float))
        rows.append(dict(pair_id=pair.pair_id, comparison_group=labels[(pair.date, pair.genus, pair.reference_group)],
                         n_common_animals=len(animals), sequence_mean_abs_gap=float(diff.abs().mean()),
                         sequence_median_abs_gap=float(diff.abs().median()),
                         sequence_min_abs_gap=float(diff.abs().min()), sequence_max_abs_gap=float(diff.abs().max()),
                         sequence_mean_signed_gap=float(diff.mean()),
                         sequence_first_median_abs_gap=float(first_diff.abs().median()),
                         sequence_same_mean_order_all_animals=bool(len(nonzero) and len(set(nonzero)) == 1),
                         sequence_same_first_order_all_animals=bool(len(set(first_sign)) == 1),
                         sequence_mean_order_majority_fraction=float(max((diff > 0).mean(), (diff < 0).mean())),
                         sequence_first_order_majority_fraction=float(max((first_diff > 0).mean(), (first_diff < 0).mean())),
                         joint_reported_n=int(joint.sum()), one_missing_n=int(one.sum()), both_missing_n=int(both_missing.sum()),
                         joint_reported_rms=float(np.sqrt(np.square(delta[joint]).mean())),
                         one_missing_energy_fraction=float(np.square(delta[one]).sum() / energy),
                         high_qcrsd_energy_fraction=float(np.square(delta[metadata.QCRSD.gt(.3)]).sum() / energy),
                         reconstructed_chemical_rms=float(np.sqrt(np.square(delta).mean()))))
    result = pairs.merge(pd.DataFrame(rows), on='pair_id', validate='one_to_one')
    assert np.allclose(result.chemical_rms_log2fc, result.reconstructed_chemical_rms, atol=1e-12, rtol=0)
    assert (result.joint_reported_n + result.one_missing_n + result.both_missing_n).eq(380).all()
    result.to_csv(OUT / 'tables/pair_context.csv', index=False)
    pd.DataFrame(animal_rows).to_csv(OUT / 'tables/pair_context_animal_positions.csv', index=False)
    groups = result.groupby(['comparison_group'] + GROUP, as_index=False).agg(
        n_pairs=('pair_id', 'size'), n_strains=('group_n_strains', 'first'), same_species_pairs=('same_species', 'sum'),
        nearest_pairs=('nearest_either', 'sum'), joint_reported_n_min=('joint_reported_n', 'min'),
        joint_reported_n_max=('joint_reported_n', 'max'), one_missing_energy_median=('one_missing_energy_fraction', 'median'),
        fixed_mean_order_pairs=('sequence_same_mean_order_all_animals', 'sum'),
        fixed_first_order_pairs=('sequence_same_first_order_all_animals', 'sum'))
    groups['has_same_and_different_species'] = groups.same_species_pairs.gt(0) & groups.same_species_pairs.lt(groups.n_pairs)
    groups.to_csv(OUT / 'tables/pair_context_groups.csv', index=False)
    summary = dict(status='success', n_pairs=len(result), n_comparison_groups=len(groups),
                   n_groups_ge3_strains=int(groups.n_strains.ge(3).sum()),
                   n_groups_with_both_species_types=int(groups.has_same_and_different_species.sum()),
                   fixed_mean_order_pairs=int(result.sequence_same_mean_order_all_animals.sum()),
                   fixed_first_order_pairs=int(result.sequence_same_first_order_all_animals.sum()),
                   n_same_species_pairs=int(result.same_species.sum()),
                   n_animals_positions=len(pd.DataFrame(animal_rows)[['date', 'worm_key']].drop_duplicates()),
                   units='chemical RMS in log2 fold-change; sequence gap in segment-index positions, not elapsed minutes',
                   species_status='exact supplied species_clean label match; no independent strain identification',
                   missing_status='original report missingness; not proven absence or concentration below a known detection limit')
    (OUT / 'logs/pair_context_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return result


def group_effects(data, value, min_pairs):
    """Descriptive effects of dependent pair graphs, computed within each graph."""
    rows = []
    covariates = ['chemical_rms_log2fc', 'joint_reported_rms', 'one_missing_energy_fraction',
                  'sequence_mean_abs_gap', 'sequence_first_median_abs_gap']
    for key, g in data.groupby('comparison_group'):
        if len(g) < min_pairs:
            continue
        common = dict(comparison_group=key, date=g.date.iloc[0], genus=g.genus.iloc[0],
                      reference_group=g.reference_group.iloc[0], n_pairs=len(g),
                      n_strains=len(set(g.strain_a) | set(g.strain_b)))
        for cov in covariates:
            if g[value].nunique() > 1 and g[cov].nunique() > 1:
                rows.append(dict(**common, effect='spearman', covariate=cov,
                                 estimate=float(spearmanr(g[value], g[cov]).statistic),
                                 n_same_species=int(g.same_species.sum()),
                                 n_different_species=int((1 - g.same_species).sum())))
        same = g.same_species.eq(1)
        if same.any() and (~same).any():
            ranks = (g[value].rank() - 1) / (len(g) - 1)
            for label, values in [('energy', g[value]), ('within_group_percentile', ranks),
                                  ('chemical_rms', g.chemical_rms_log2fc),
                                  ('sequence_gap', g.sequence_mean_abs_gap)]:
                rows.append(dict(**common, effect='same_minus_different_species', covariate=label,
                                 estimate=float(values[same].mean() - values[~same].mean()),
                                 n_same_species=int(same.sum()), n_different_species=int((~same).sum())))
    return pd.DataFrame(rows)


def summarize_effects(effects):
    if effects.empty:
        return pd.DataFrame()
    return effects.groupby(['effect', 'covariate'], as_index=False).agg(
        equal_group_estimate=('estimate', 'mean'), n_groups=('estimate', 'size'),
        n_positive_groups=('estimate', lambda x: int(x.gt(0).sum())),
        n_negative_groups=('estimate', lambda x: int(x.lt(0).sum())),
        min_group_estimate=('estimate', 'min'), max_group_estimate=('estimate', 'max'))


def patterns(context_data):
    path = OUT / 'tables/pair_signal_summary.csv'
    if not path.exists():
        print('Neural summary not yet available: context-only run completed.')
        return
    signal = pd.read_csv(path, dtype={'date': str})
    columns = ['pair_id', 'energy_fixed', 'energy_common10_fixed', 'all_13_cells']
    data = context_data.merge(signal[columns], on='pair_id', validate='one_to_one')
    assert len(data) == 147
    assert data.all_13_cells.dtype == bool and int(data.all_13_cells.sum()) == 137
    all_effects, summaries, deletions = [], [], []
    specifications = [('available_cells_147', data, 'energy_fixed'),
                      ('complete13_137', data[data.all_13_cells], 'energy_fixed'),
                      ('common10_147', data, 'energy_common10_fixed')]
    for label, d, value in specifications:
        for minimum in [3, 6]:
            effects = group_effects(d, value, minimum)
            effects['neural_coverage'] = label
            effects['min_pairs_per_group'] = minimum
            all_effects.append(effects)
            summary = summarize_effects(effects)
            summary['neural_coverage'] = label
            summary['min_pairs_per_group'] = minimum
            summaries.append(summary)
            # Remove each strain everywhere it appears, or a complete shared
            # recording block. Recompute group effects; these are influence
            # ranges, not sampling intervals or independent replications.
            for deletion_type, names in [('strain', sorted(set(d.strain_a) | set(d.strain_b))),
                                         ('block', sorted(d.date.unique()))]:
                for name in names:
                    keep = (~d.strain_a.eq(name) & ~d.strain_b.eq(name)) if deletion_type == 'strain' else ~d.date.eq(name)
                    deleted = summarize_effects(group_effects(d[keep], value, minimum))
                    deleted['neural_coverage'] = label
                    deleted['min_pairs_per_group'] = minimum
                    deleted['deletion_type'] = deletion_type
                    deleted['deleted'] = name
                    deletions.append(deleted)
    groups = pd.concat(all_effects, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    deletion = pd.concat(deletions, ignore_index=True)
    index = ['neural_coverage', 'min_pairs_per_group', 'effect', 'covariate']
    for deletion_type in ['strain', 'block']:
        ranges = deletion[deletion.deletion_type.eq(deletion_type)].groupby(index).agg(
            estimate_min=('equal_group_estimate', 'min'), estimate_max=('equal_group_estimate', 'max'),
            groups_min=('n_groups', 'min'), groups_max=('n_groups', 'max'))
        ranges = ranges.add_prefix(f'delete_{deletion_type}_').reset_index()
        summary = summary.merge(ranges, on=index, validate='one_to_one')
    groups.to_csv(OUT / 'tables/pair_context_pattern_groups.csv', index=False)
    summary.to_csv(OUT / 'tables/pair_context_pattern_summary.csv', index=False)
    deletion.to_csv(OUT / 'tables/pair_context_pattern_deletions.csv', index=False)
    checks = json.loads((OUT / 'logs/pair_context_verification.json').read_text())
    checks.update(pattern_status='success',
                  neural_coverage_rules=['147 available-cell means', '137 complete 13-cell pairs', '147 shared 10-cell means'],
                  group_rules='At least 3 pairs for primary descriptive directions; at least 6 pairs sensitivity',
                  outcome_selection='No thresholds, cells, or chemical features selected from neural outcome',
                  n_group_effect_rows=len(groups), n_pattern_summary_rows=len(summary),
                  n_influence_rows=len(deletion),
                  inference='No pair-independent p-values or sampling confidence intervals; deletion ranges are sensitivity only')
    (OUT / 'logs/pair_context_verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    primary = summary[summary.neural_coverage.eq('available_cells_147') & summary.min_pairs_per_group.eq(3)]
    print(primary.to_string(index=False))


if __name__ == '__main__':
    patterns(context())

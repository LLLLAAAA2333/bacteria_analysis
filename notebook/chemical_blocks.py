"""Exploratory, resampling-supported chemical blocks and equal-block RMS distance.

Rows are bacterial reference profiles, not replicate LC-MS measurements.
Resampling rows assesses sensitivity to strain composition; it does not estimate
experimental measurement uncertainty or establish independent biological replicates.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


def spearman_matrix(values):
    """Column correlations; constant columns have zero association with other columns."""
    ranks = rankdata(values, axis=0)
    ranks -= ranks.mean(axis=0)
    norms = np.linalg.norm(ranks, axis=0)
    normalized = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)
    correlation = np.clip(normalized.T @ normalized, -1, 1)
    np.fill_diagonal(correlation, 1)
    return correlation


def complete_groups(dissimilarity, threshold):
    if len(dissimilarity) == 1:
        return np.ones(1, dtype=int)
    dissimilarity = np.maximum((dissimilarity + dissimilarity.T)/2, 0)
    np.fill_diagonal(dissimilarity, 0)
    tree = linkage(squareform(dissimilarity, checks=False), method='complete')
    labels = fcluster(tree, t=threshold+1e-12, criterion='distance')
    # Group names are ordered by the first feature in the original column order.
    return pd.factorize(labels, sort=False)[0]+1


def equal_block_distance(profiles, groups):
    """Compute cell34 exactly, retaining all molecules and their signed log2FC."""
    if not profiles.columns.equals(groups.index):
        raise ValueError('Group assignments must align exactly with profile columns')
    if groups.isna().any():
        raise ValueError('Every feature needs exactly one group, including singletons')
    sizes = groups.value_counts()
    count = len(sizes)
    weights = pd.Series(1/(count * groups.map(sizes).to_numpy(float)),
                        index=profiles.columns, name='block_feature_weight')
    values = profiles.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Profiles must be finite; do not drop features pairwise')
    distances = squareform(pdist(values * np.sqrt(weights.to_numpy()), metric='euclidean'))
    rdm = pd.DataFrame(distances, index=profiles.index, columns=profiles.index)
    if not np.isclose(weights.sum(), 1):
        raise RuntimeError('Feature weights must sum to one')
    return rdm, weights


def build_chemical_blocks(profiles, baseline_rdm, correlation_threshold=0.7,
                          stability_threshold=0.8, n_resamples=200, seed=20260920,
                          absolute_correlation=True):
    """Infer disjoint blocks, require all within-block pairs to meet both thresholds.

    Bootstrap coassignment is the frequency of two molecules appearing in the
    same complete-linkage block after resampling bacterial rows. Recompute ranks
    in every resample. Final blocks satisfy BOTH the full-data correlation and
    coassignment thresholds. Thresholds are exploratory choices, not significance tests.
    """
    from threadpoolctl import threadpool_limits

    if profiles.shape[0] < 3 or profiles.shape[1] < 1:
        raise ValueError('Need at least three bacterial profiles and one feature')
    if not profiles.index.is_unique or not profiles.columns.is_unique:
        raise ValueError('Sample and feature labels must be unique')
    if not profiles.index.equals(baseline_rdm.index) or not baseline_rdm.index.equals(baseline_rdm.columns):
        raise ValueError('chemical_profiles and chemical_rdm must contain the same ordered sample IDs')
    if not 0 < correlation_threshold < 1 or not 0 < stability_threshold < 1:
        raise ValueError('Correlation and stability thresholds must be between 0 and 1')
    if not isinstance(n_resamples, int) or n_resamples < 20:
        raise ValueError('Use an integer n_resamples >= 20')
    values = profiles.to_numpy(float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError('Use the fixed finite log2FC feature set from cell9')
    n, p = values.shape
    ordinary = squareform(pdist(values))/np.sqrt(p)
    if not np.allclose(ordinary, baseline_rdm.to_numpy(), rtol=1e-10, atol=1e-12):
        raise ValueError('Baseline RDM differs from RMS log2FC profiles; rerun cell9')
    if not np.any(ordinary > 0):
        raise ValueError('All profiles are identical; distance comparisons are undefined')

    def similarity(matrix):
        return np.abs(matrix) if absolute_correlation else matrix

    with threadpool_limits(limits=1, user_api='blas'):
        correlation = spearman_matrix(values)
        association = similarity(correlation)
        full_labels = complete_groups(1-association, 1-correlation_threshold)
        counts = np.zeros((p, p), dtype=np.int64)
        rng = np.random.default_rng(seed)
        for iteration in range(n_resamples):
            selected = rng.integers(0, n, size=n)
            resampled = similarity(spearman_matrix(values[selected]))
            labels = complete_groups(1-resampled, 1-correlation_threshold)
            counts += labels[:, None] == labels[None, :]
            if (iteration+1) % 50 == 0 or iteration+1 == n_resamples:
                print(f'Chemical block composition resampling: {iteration+1}/{n_resamples}', flush=True)
    support = counts/n_resamples
    # Complete linkage prevents a chain of weakly associated molecules forming a block.
    consensus_distance = np.maximum((1-association)/(1-correlation_threshold),
                                    (1-support)/(1-stability_threshold))
    final_labels = complete_groups(consensus_distance, 1)
    groups = pd.Series([f'C{label:03d}' for label in final_labels], index=profiles.columns, name='block')
    block_rdm, weights = equal_block_distance(profiles, groups)
    constants = np.ptp(values, axis=0) == 0
    feature_variance = profiles.var(axis=0, ddof=1)
    members = pd.DataFrame(dict(block=groups, original_feature_weight=1/p,
                                block_feature_weight=weights, weight_ratio=weights*p,
                                constant_profile=constants))
    members.index.name = 'metabolite'
    table = []
    for name in groups.unique():
        indices = np.flatnonzero(groups.to_numpy() == name)
        k = len(indices)
        ii, jj = np.triu_indices(k, 1)
        pair_association = association[np.ix_(indices, indices)][ii, jj]
        pair_support = support[np.ix_(indices, indices)][ii, jj]
        if k > 1 and (pair_association.min() < correlation_threshold-1e-12 or pair_support.min() < stability_threshold-1e-12):
            raise RuntimeError('A final block violates its pairwise criteria')
        # Mean over unordered sample pairs of (L_i-L_j)^2 is twice sample variance.
        average_squared = 2*feature_variance.iloc[indices]
        table.append(dict(block=name, n_features=k, singleton=k==1,
                          min_association=float(pair_association.min()) if k>1 else np.nan,
                          min_coassignment=float(pair_support.min()) if k>1 else np.nan,
                          mean_coassignment=float(pair_support.mean()) if k>1 else np.nan,
                          original_total_weight=k/p, block_total_weight=1/groups.nunique(),
                          per_feature_weight_ratio=p/(groups.nunique()*k),
                          original_mean_squared_contribution=float(average_squared.sum()/p),
                          block_mean_squared_contribution=float(average_squared.mean()/groups.nunique())))
    group_table = pd.DataFrame(table).set_index('block')
    for prefix in ('original', 'block'):
        column = f'{prefix}_mean_squared_contribution'
        group_table[f'{prefix}_contribution_share'] = group_table[column]/group_table[column].sum()
    upper = np.triu_indices(n, 1)
    original_pairs, block_pairs = ordinary[upper], block_rdm.to_numpy()[upper]
    align_scale = float(original_pairs @ block_pairs/(block_pairs @ block_pairs))
    # Descriptive agreement only: pairs share bacterial samples and are not independent observations.
    from scipy.stats import spearmanr
    metrics = dict(n_samples=n, n_features=p, n_pairs=len(original_pairs),
                   initial_groups=len(np.unique(full_labels)), n_groups=groups.nunique(),
                   multi_feature_groups=int((group_table.n_features>1).sum()),
                   singleton_groups=int(group_table.singleton.sum()), constant_features=int(constants.sum()),
                   singleton_total_weight=float(group_table.singleton.mean()),
                   largest_group=int(group_table.n_features.max()),
                   distance_spearman=float(spearmanr(original_pairs, block_pairs).statistic),
                   distance_pearson=float(np.corrcoef(original_pairs, block_pairs)[0, 1]),
                   raw_relative_change=float(np.linalg.norm(block_pairs-original_pairs)/np.linalg.norm(original_pairs)),
                   align_scale=align_scale,
                   scale_aligned_relative_change=float(np.linalg.norm(align_scale*block_pairs-original_pairs)/np.linalg.norm(original_pairs)))
    parameters = dict(correlation='absolute Spearman' if absolute_correlation else 'positive Spearman',
                      correlation_threshold=correlation_threshold, stability_threshold=stability_threshold,
                      n_resamples=n_resamples, seed=seed, linkage='complete',
                      resampling_unit='bacterial profile row; sensitivity to strain composition, not measurement error',
                      distance_values='original signed log2FC; no z-scoring or group-mean profiles',
                      unassigned_features='one singleton block per feature')
    block_rdm.attrs.update(baseline_rdm.attrs)
    block_rdm.attrs.update(distance='Group-equal RMS difference of log2FC', n_groups=groups.nunique(),
                           feature_weighting='1 / (number of groups * number of features in own group)',
                           group_parameters=parameters, group_validation='Exploratory strain-composition resampling only')
    return dict(rdm=block_rdm, baseline_rdm=baseline_rdm.copy(), groups=groups, members=members,
                group_summary=group_table, metrics=metrics, parameters=parameters,
                correlation=pd.DataFrame(correlation, index=profiles.columns, columns=profiles.columns),
                coassignment=pd.DataFrame(support, index=profiles.columns, columns=profiles.columns),
                input_profiles=profiles.copy())


def check_blocks_in_subset(full_result, sample_ids, n_resamples=1000, seed=20260921):
    """Check full-library relationships in a subset without redefining blocks.

    Paired support is the frequency of maintaining the FULL-library correlation
    direction AND meeting the correlation-strength threshold after resampling
    subset rows. It is not coassignment frequency: the subset is never reclustered.
    Because the subset is part of discovery data, this is an applicability check,
    not independent validation or a neural-association significance test.
    """
    from threadpoolctl import threadpool_limits

    profiles = full_result['input_profiles']
    sample_ids = pd.Index(sample_ids)
    if not sample_ids.is_unique or not sample_ids.isin(profiles.index).all():
        raise ValueError('Subset IDs must be unique and present in the full chemical library')
    if len(sample_ids) < 3:
        raise ValueError('At least three paired samples are needed to check correlations')
    if not isinstance(n_resamples, int) or n_resamples < 20:
        raise ValueError('Use an integer n_resamples >= 20')
    subset = profiles.loc[sample_ids]
    values = subset.to_numpy(float)
    groups = full_result['groups']
    if not groups.index.equals(profiles.columns):
        raise ValueError('Full-library group assignments and features are misaligned')
    parameters = full_result['parameters']
    threshold = parameters['correlation_threshold']
    stability = parameters['stability_threshold']
    absolute = parameters['correlation'] == 'absolute Spearman'
    rows = []
    for group in groups.unique():
        members = np.flatnonzero(groups.to_numpy() == group)
        ii, jj = np.triu_indices(len(members), 1)
        rows.extend((group, int(members[i]), int(members[j])) for i, j in zip(ii, jj))
    pairs = pd.DataFrame(rows, columns=['block', 'feature_i', 'feature_j'])
    pi, pj = pairs.feature_i.to_numpy(int), pairs.feature_j.to_numpy(int)
    full_correlation = full_result['correlation'].to_numpy()[pi, pj]
    correlation = spearman_matrix(values)
    variable = np.ptp(values, axis=0) > 0
    defined = variable[pi] & variable[pj]
    subset_correlation = correlation[pi, pj]
    association = np.abs(subset_correlation) if absolute else subset_correlation
    same_sign = defined & (np.sign(subset_correlation) == np.sign(full_correlation))
    strength_pass = defined & (association >= threshold-1e-12)
    support_counts = np.zeros(len(pairs), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(pairs):
        with threadpool_limits(limits=1, user_api='blas'):
            for iteration in range(n_resamples):
                selected = rng.integers(0, len(subset), size=len(subset))
                sampled = values[selected]
                draw = spearman_matrix(sampled)[pi, pj]
                draw_variable = np.ptp(sampled, axis=0) > 0
                valid = draw_variable[pi] & draw_variable[pj]
                draw_association = np.abs(draw) if absolute else draw
                support_counts += (valid & (draw_association >= threshold-1e-12)
                                   & (np.sign(draw) == np.sign(full_correlation)))
                if (iteration+1) % 100 == 0 or iteration+1 == n_resamples:
                    print(f'Paired chemical relationship check: {iteration+1}/{n_resamples}', flush=True)
    support = support_counts/n_resamples
    pairs['metabolite_i'] = profiles.columns.to_numpy()[pi]
    pairs['metabolite_j'] = profiles.columns.to_numpy()[pj]
    pairs['full_spearman'] = full_correlation
    pairs['paired_spearman'] = np.where(defined, subset_correlation, np.nan)
    pairs['paired_association'] = np.where(defined, association, np.nan)
    pairs['paired_defined'] = defined
    pairs['paired_same_sign'] = same_sign
    pairs['paired_strength_pass'] = strength_pass
    pairs['paired_relation_support'] = support
    pairs['paired_relation_pass'] = same_sign & strength_pass & (support >= stability-1e-12)
    summary = []
    for group in groups.unique():
        local = pairs.loc[pairs.block == group]
        if local.empty:
            summary.append(dict(block=group, within_block_pairs=0,
                                paired_min_association=np.nan, paired_min_relation_support=np.nan,
                                paired_all_signs_agree=pd.NA, paired_relationships_pass=pd.NA,
                                paired_check_status='singleton: no within-block relationship'))
        else:
            passes = bool(local.paired_relation_pass.all())
            summary.append(dict(block=group, within_block_pairs=len(local),
                                paired_min_association=float(local.paired_association.min())
                                if local.paired_defined.all() else np.nan,
                                paired_min_relation_support=float(local.paired_relation_support.min()),
                                paired_all_signs_agree=bool(local.paired_same_sign.all()),
                                paired_relationships_pass=passes,
                                paired_check_status='passes specified criteria' if passes else 'does not meet all criteria'))
    summary = pd.DataFrame(summary).set_index('block')
    for column in ('paired_all_signs_agree', 'paired_relationships_pass'):
        summary[column] = summary[column].astype('boolean')
    multi = summary.within_block_pairs > 0
    metrics = dict(n_full_samples=len(profiles), n_paired_samples=len(subset),
                   n_features=len(profiles.columns), within_block_pairs=len(pairs),
                   multi_feature_groups=int(multi.sum()),
                   groups_passing_subset_check=int(summary.loc[multi, 'paired_relationships_pass'].sum()),
                   groups_not_passing_subset_check=int((~summary.loc[multi, 'paired_relationships_pass']).sum()),
                   pairs_with_sign_reversal=int((defined & ~same_sign).sum()),
                   pairs_undefined_in_subset=int((~defined).sum()),
                   pairs_passing_subset_check=int(pairs.paired_relation_pass.sum()))
    # Direct subsetting freezes full-library groups, molecules and weights exactly.
    rdm = full_result['rdm'].loc[sample_ids, sample_ids].copy()
    baseline = full_result['baseline_rdm'].loc[sample_ids, sample_ids].copy()
    for matrix in (rdm, baseline):
        matrix.attrs.update(sample_ids=sample_ids.tolist(), sample_scope='Neural/chemical ID intersection',
                            group_discovery_scope='All bacterial rows in chemical matrix',
                            feature_selection_scope='Full chemical library; fixed in paired subset')
    return dict(profiles=subset.copy(), rdm=rdm, baseline_rdm=baseline, pairs=pairs,
                group_summary=summary, metrics=metrics,
                correlation=pd.DataFrame(correlation, index=profiles.columns, columns=profiles.columns),
                parameters=dict(n_resamples=n_resamples, seed=seed, correlation_threshold=threshold,
                                support_threshold=stability, full_groups_frozen=True,
                                support_definition='Frequency of retaining full-library sign and meeting strength threshold',
                                interpretation='Subset applicability check; overlaps discovery data, not independent validation'))

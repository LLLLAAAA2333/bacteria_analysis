"""Describe all four chemical/neural distance corners using saved 106-sample data.

Notebook-callable: run_exploration(repo, out, tail_fraction=0.25).
No response refitting, new bootstrap, imputation, significance test, or raw writes.
Chemical differences are log2FC differences, not concentration differences.
"""
from pathlib import Path
import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd

CATEGORIES = ('Cnear_Nnear', 'Cnear_Nfar', 'Cfar_Nnear', 'Cfar_Nfar')


def classify(chemical, neural, q):
    if not 0 < q < .5:
        raise ValueError('tail_fraction must be between zero and one half')
    c0, c1 = np.quantile(chemical, [q, 1-q])
    n0, n1 = np.quantile(neural, [q, 1-q])
    group = np.full(len(chemical), 'middle', dtype=object)
    for c, cm in [('near', chemical <= c0), ('far', chemical >= c1)]:
        for n, nm in [('near', neural <= n0), ('far', neural >= n1)]:
            group[cm & nm] = f'C{c}_N{n}'
    return group, dict(chemical_near=float(c0), chemical_far=float(c1),
                       neural_near=float(n0), neural_far=float(n1))


def endpoint_weights(a, b):
    """Average over each sample's partners, then equally over participating samples.

    This is a descriptive hub-sensitivity check, not independent observations.
    """
    counts = pd.Series(np.r_[a, b]).value_counts()
    weights = np.array([1/counts[x] + 1/counts[y] for x, y in zip(a, b)])
    return weights / weights.sum()


def run_exploration(repo, out, tail_fraction=.25):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    current = repo / 'reports/exploration_response_profiles_individual_snr_20261002'
    chem_dir = repo / 'reports/population_first_20260930/tables'
    inputs = [current/'tables/rdm_matched_pairs.csv', current/'tables/strain_coefficients.csv',
              current/'tables/condition_metrics.csv', current/'tables/hmds_neural_valid_fraction.csv',
              chem_dir/'aligned_chemical_log2fc_paired.csv', chem_dir/'aligned_chemical_metadata.csv',
              chem_dir/'aligned_chemical_report_observed_paired.parquet',
              chem_dir/'aligned_chemical_reference_groups_paired.csv',
              chem_dir/'aligned_taxonomy_paired.csv']
    protected = inputs + list((repo/'notebook').glob('*.ipynb'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    pairs = pd.read_csv(inputs[0]).rename(columns={'filtered': 'neural'})
    coef = pd.read_csv(inputs[1], index_col=0)
    metrics = pd.read_csv(inputs[2])
    coverage = pd.read_csv(inputs[3], index_col=0).loc[coef.index, coef.index]
    chemical = pd.read_csv(inputs[4], index_col=0).loc[coef.index]
    meta = pd.read_csv(inputs[5]).set_index('metabolite').loc[chemical.columns]
    observed = pd.read_parquet(inputs[6]).loc[coef.index, chemical.columns]
    reference = pd.read_csv(inputs[7], index_col=0).reference_group
    taxonomy = pd.read_csv(inputs[8], index_col=0)
    assert chemical.shape == (106, 380) and coef.shape == (106, 13)
    assert chemical.index.is_unique and chemical.columns.is_unique
    assert np.isfinite(chemical.to_numpy()).all() and np.isfinite(coef.to_numpy()).all()
    assert not observed.isna().any().any()
    observed = observed.to_numpy(bool)
    ai = coef.index.get_indexer(pairs.strain_a)
    bi = coef.index.get_indexer(pairs.strain_b)
    assert (ai >= 0).all() and (bi >= 0).all() and (ai < bi).all()
    assert len(pairs) == 106*105//2 and not pairs.duplicated(['strain_a','strain_b']).any()
    x = chemical.to_numpy()
    delta = np.abs(x[ai] - x[bi])
    sq = delta**2
    rms = np.sqrt(sq.mean(axis=1))
    unit = coef.to_numpy() / np.linalg.norm(coef.to_numpy(), axis=1)[:, None]
    neural_contribution = .5*(unit[ai]-unit[bi])**2
    neural_rebuilt = neural_contribution.sum(axis=1)
    assert np.allclose(rms, pairs.chemical, rtol=1e-12, atol=1e-12)
    assert np.allclose(neural_rebuilt, pairs.neural, rtol=1e-12, atol=1e-12)
    # Each row sums to 380; mean across features is one for every pair.
    relative = sq / sq.mean(axis=1)[:, None]
    both = observed[ai] & observed[bi]
    pairs.insert(0, 'pair_id', pairs.strain_a + '__' + pairs.strain_b)
    groups, thresholds = classify(pairs.chemical.to_numpy(), pairs.neural.to_numpy(), tail_fraction)
    pairs['category'] = groups
    pairs['chemical_percentile'] = pairs.chemical.rank(pct=True)
    pairs['neural_percentile'] = pairs.neural.rank(pct=True)
    pairs['bootstrap_valid_fraction'] = coverage.to_numpy()[ai, bi]
    pairs['reference_a'] = pairs.strain_a.map(reference)
    pairs['reference_b'] = pairs.strain_b.map(reference)
    pairs['same_reference'] = pairs.reference_a.eq(pairs.reference_b)
    date_sets = metrics.groupby('strain').block.apply(lambda s: set(s.astype(str)))
    pairs['shares_neural_date'] = [bool(date_sets[a] & date_sets[b]) for a,b in zip(pairs.strain_a,pairs.strain_b)]
    pairs['same_genus'] = pairs.strain_a.map(taxonomy.genus_clean).eq(pairs.strain_b.map(taxonomy.genus_clean))
    for method in ['unfiltered','raw','snr_0.25','snr_0.75','snr_1']:
        alt, _ = classify(pairs.chemical.to_numpy(), pairs[method].to_numpy(), tail_fraction)
        pairs['category_'+method] = alt
    def clean_annotation(value):
        value = unicodedata.normalize('NFKC', str(value)).strip()
        return 'Unannotated' if value.lower() in ('', 'na', 'nan') else value
    meta['superclass'] = meta.SuperClass.map(clean_annotation)
    meta.to_csv(tables/'feature_metadata.csv', index_label='feature')
    pairs.to_csv(tables/'pair_catalogue.csv', index=False)
    np.savez_compressed(tables/'pair_arrays.npz', abs_chemical_difference=delta,
                        chemical_relative_contribution=relative,
                        neural_contribution=neural_contribution,
                        both_reported=both, features=chemical.columns.to_numpy(str),
                        cells=coef.columns.to_numpy(str), pair_ids=pairs.pair_id.to_numpy(str))
    feature_rows, class_rows, summaries, degree_rows, cell_rows = [], [], [], [], []
    for category in CATEGORIES:
        use = groups == category
        z = pairs.loc[use]
        weights = endpoint_weights(z.strain_a.to_numpy(), z.strain_b.to_numpy())
        degree = pd.Series(np.r_[z.strain_a, z.strain_b]).value_counts()
        summaries.append(dict(category=category,n_pairs=len(z),n_samples=len(degree),
            median_chemical=float(z.chemical.median()),median_neural=float(z.neural.median()),
            same_reference_fraction=float(z.same_reference.mean()),
            shares_neural_date_fraction=float(z.shares_neural_date.mean()),
            same_genus_fraction=float(z.same_genus.mean()),
            median_bootstrap_valid_fraction=float(z.bootstrap_valid_fraction.median()),
            below_half_bootstrap_fraction=float((z.bootstrap_valid_fraction < .5).mean()),
            unfiltered_category_retention=float(z.category_unfiltered.eq(category).mean()),
            raw_category_retention=float(z.category_raw.eq(category).mean()),
            top_sample=str(degree.index[0]),top_sample_pair_fraction=float(degree.iloc[0]/len(z))))
        for s, count in degree.items():
            degree_rows.append(dict(category=category,sample_id=s,n_pairs=int(count),
                                   genus=taxonomy.loc[s,'genus_clean'],reference=reference.loc[s]))
        for k, feature in enumerate(chemical.columns):
            valid = both[use,k]
            feature_rows.append(dict(category=category,feature=feature,
                superclass=meta.loc[feature,'superclass'],qc_rsd=meta.loc[feature,'QCRSD'],
                complete_162_eligible=bool(meta.loc[feature,'previous_complete_162_eligible']),
                mean_abs_log2fc_difference=float(delta[use,k].mean()),
                mean_relative_contribution=float(relative[use,k].mean()),
                mean_distance_share_pct=float(relative[use,k].mean()/380*100),
                endpoint_balanced_relative_contribution=float(weights @ relative[use,k]),
                both_reported_fraction=float(valid.mean()),
                mean_abs_both_reported=float(delta[use,k][valid].mean()) if valid.any() else np.nan))
        for superclass, frame in meta.groupby('superclass', sort=True):
            cols = chemical.columns.get_indexer(frame.index)
            class_rows.append(dict(category=category,superclass=superclass,n_features=len(cols),
                mean_relative_contribution=float(relative[use][:,cols].mean()),
                mean_distance_share_pct=float(relative[use][:,cols].sum(axis=1).mean()/380*100),
                endpoint_balanced_relative_contribution=float(weights @ relative[use][:,cols].mean(axis=1)),
                both_reported_fraction=float(both[use][:,cols].mean())))
        for k, cell in enumerate(coef.columns):
            cell_rows.append(dict(category=category,cell=cell,
                mean_cosine_distance_contribution=float(neural_contribution[use,k].mean()),
                mean_distance_share_pct=float((neural_contribution[use,k]/pairs.loc[use,'neural']).mean()*100)))
    feature_summary = pd.DataFrame(feature_rows)
    class_summary = pd.DataFrame(class_rows)
    pd.DataFrame(summaries).to_csv(tables/'category_summary.csv', index=False)
    pd.DataFrame(degree_rows).to_csv(tables/'sample_pair_degrees.csv', index=False)
    feature_summary.to_csv(tables/'feature_summary.csv', index=False)
    class_summary.to_csv(tables/'class_summary.csv', index=False)
    pd.DataFrame(cell_rows).to_csv(tables/'neural_cell_summary.csv', index=False)
    # Display selection only: retain the full 380-feature distance definition.
    # Use pre-existing complete-report/QC <= .30 list for named highlights.
    eligible = meta.index[meta.previous_complete_162_eligible.eq(True)]
    selected = []
    for category in CATEGORIES:
        top = feature_summary.query('category == @category').set_index('feature').loc[eligible]
        selected.extend(top.mean_relative_contribution.nlargest(3).index.tolist())
    feature_pivot = feature_summary.pivot(index='feature',columns='category',values='mean_relative_contribution')
    contrasts = []
    for band in ['near','far']:
        effect = np.log2(feature_pivot[f'C{band}_Nfar']/feature_pivot[f'C{band}_Nnear'])
        selected.extend(effect.loc[eligible].abs().nlargest(4).index.tolist())
        for f,v in effect.items():
            contrasts.append(dict(chemical_band=band,feature=f,log2_relative_contribution_ratio=float(v),
                                   complete_162_eligible=bool(f in eligible)))
    selected = list(dict.fromkeys(selected))
    pd.DataFrame(dict(feature=selected,superclass=meta.loc[selected,'superclass'].to_numpy(),
                      selection_rank=np.arange(1,len(selected)+1))).to_csv(tables/'selected_features.csv', index=False)
    pd.DataFrame(contrasts).to_csv(tables/'feature_contrasts.csv', index=False)
    sensitivity, direction_checks = [], []
    for q in [.2,.25,.3]:
        alt, cut = classify(pairs.chemical.to_numpy(),pairs.neural.to_numpy(),q)
        for category in CATEGORIES:
            z = pairs.loc[alt == category]
            sensitivity.append(dict(tail_fraction=q,category=category,n_pairs=len(z),
                                    n_samples=len(set(z.strain_a)|set(z.strain_b)),**cut))
        for band in ['near','far']:
            near = relative[alt == f'C{band}_Nnear'].mean(axis=0)
            far = relative[alt == f'C{band}_Nfar'].mean(axis=0)
            for f,v in zip(chemical.columns,np.log2(far/near)):
                direction_checks.append(dict(tail_fraction=q,chemical_band=band,feature=f,
                                             log2_relative_contribution_ratio=float(v)))
    pd.DataFrame(sensitivity).to_csv(tables/'threshold_sensitivity.csv',index=False)
    pd.DataFrame(direction_checks).to_csv(tables/'feature_threshold_sensitivity.csv',index=False)
    # Contributions from report missingness and high QC RSD may overlap; do not
    # add these percentages. Neither mask changes the existing primary RDM.
    qc_rows = []
    for category in CATEGORIES:
        use = groups == category
        masks = {'both_reported':both,
                 'one_sided_report_missing':observed[ai] ^ observed[bi],
                 'both_report_missing':~observed[ai] & ~observed[bi],
                 'qc_rsd_above_0.30':np.broadcast_to(meta.QCRSD.to_numpy()>.3,both.shape),
                 'complete_162':np.broadcast_to(meta.previous_complete_162_eligible.to_numpy(bool),both.shape)}
        for name, mask in masks.items():
            share = (relative[use] * mask[use]).sum(axis=1)/380*100
            qc_rows.append(dict(category=category,feature_scope=name,
                                mean_distance_share_pct=float(share.mean())))
    pd.DataFrame(qc_rows).to_csv(tables/'chemical_distance_quality.csv',index=False)
    # Fixed original cutoffs for these contextual checks: no reclassification.
    # Reference-stratified contrasts are unavailable if either corner is empty.
    contrast_checks, strata = [], []
    complete_cols = meta.previous_complete_162_eligible.to_numpy(bool)
    complete_relative = sq / sq[:,complete_cols].mean(axis=1)[:,None]
    def positive_log_ratio(numerator, denominator):
        good = (numerator > 0) & (denominator > 0)
        result = np.full(np.shape(numerator),np.nan)
        result[good] = np.log2(numerator[good]/denominator[good])
        return result
    for band in ['near','far']:
        near_mask, far_mask = groups == f'C{band}_Nnear', groups == f'C{band}_Nfar'
        for reference_group in ['all','same_reference',*sorted(reference.unique())]:
            if reference_group == 'all':
                context = np.ones(len(pairs),bool)
            elif reference_group == 'same_reference':
                context = pairs.same_reference.to_numpy()
            else:
                context = (pairs.reference_a.eq(reference_group)&pairs.reference_b.eq(reference_group)).to_numpy()
            na, fa = near_mask & context, far_mask & context
            strata.append(dict(chemical_band=band,reference_scope=reference_group,
                               n_neural_near=int(na.sum()),n_neural_far=int(fa.sum())))
            if not na.any() or not fa.any():
                continue
            near_mean, far_mean = relative[na].mean(axis=0),relative[fa].mean(axis=0)
            ratio = positive_log_ratio(far_mean,near_mean)
            absolute_ratio = positive_log_ratio(delta[fa].mean(axis=0),delta[na].mean(axis=0))
            complete_ratio = positive_log_ratio(complete_relative[fa].mean(axis=0),complete_relative[na].mean(axis=0))
            for k,f in enumerate(chemical.columns):
                contrast_checks.append(dict(chemical_band=band,reference_scope=reference_group,
                    feature=f,log2_relative_contribution_ratio=float(ratio[k]),
                    near_mean_relative_contribution=float(near_mean[k]),
                    far_mean_relative_contribution=float(far_mean[k]),
                    log2_mean_absolute_difference_ratio=float(absolute_ratio[k]),
                    log2_complete162_normalized_contribution_ratio=float(complete_ratio[k]),
                    n_neural_near=int(na.sum()),n_neural_far=int(fa.sum())))
        # Recompute pair means after dropping every edge touching one sample.
        deletions = []
        for sample in coef.index:
            keep = (pairs.strain_a.ne(sample) & pairs.strain_b.ne(sample)).to_numpy()
            na, fa = near_mask & keep, far_mask & keep
            deletions.append(np.log2(relative[fa].mean(axis=0)/relative[na].mean(axis=0)))
        deletions = np.asarray(deletions)
        for k,f in enumerate(chemical.columns):
            contrast_checks.append(dict(chemical_band=band,reference_scope='leave_one_sample_range',
                feature=f,log2_relative_contribution_ratio=np.nan,
                leave_one_min=float(deletions[:,k].min()),leave_one_max=float(deletions[:,k].max()),
                n_neural_near=int(near_mask.sum()),n_neural_far=int(far_mask.sum())))
    pd.DataFrame(strata).to_csv(tables/'reference_stratum_counts.csv',index=False)
    pd.DataFrame(contrast_checks).to_csv(tables/'feature_context_checks.csv',index=False)
    # Complete-feature inspection preserves names/order, and native log2FC units.
    chemical.to_csv(tables/'sample_chemical_log2fc.csv')
    coef.to_csv(tables/'sample_neural_coefficients.csv')
    parameters = dict(tail_fraction=tail_fraction,primary_thresholds=thresholds,
        categories=list(CATEGORIES),n_samples=len(coef),n_pairs=len(pairs),n_features=380,n_cells=13,
        distance_neural='1 - cosine of existing individual-SNR>=0.5 signed template coefficients',
        distance_chemical='RMS difference of existing 380 log2FC features',
        normalized_feature_contribution='delta_f^2 / mean_over_380_features(delta^2); row mean = 1',
        neural_contribution='0.5 * (unit_vector_a - unit_vector_b)^2 per cell; sum = 1-cosine',
        feature_highlights='Union of top 3 contributions per category and top 4 absolute log2 far/near neural ratios within each chemical band, restricted to pre-existing complete-report/QC<=0.30 162-feature list; descriptive selection on same data',
        pair_weighting='Equal unordered pairs; endpoint-balanced sensitivity averages partner pairs within sample then samples equally',
        scope='Exploration of four distance corners only; no mechanistic interpretation or technology-performance claim',
        source_sha256=hashes)
    (out/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    checks = dict(n_pairs=len(pairs),category_counts={c:int((groups==c).sum()) for c in (*CATEGORIES,'middle')},
        chemical_distance_max_error=float(np.max(np.abs(rms-pairs.chemical))),
        neural_distance_max_error=float(np.max(np.abs(neural_rebuilt-pairs.neural))),
        normalized_contribution_row_mean_max_error=float(np.max(np.abs(relative.mean(axis=1)-1))),
        n_highlight_features=len(selected),n_highlight_eligible_features=len(eligible),
        primary_chemical_reference_groups=reference.value_counts().to_dict(),
        inputs_and_notebooks_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
    assert checks['inputs_and_notebooks_unchanged']
    (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks


if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1]
    print(json.dumps(run_exploration(destination.parents[1],destination),indent=2))

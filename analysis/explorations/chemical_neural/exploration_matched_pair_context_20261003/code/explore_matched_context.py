"""Inspect chemical-distance-matched comparisons under strict sample context.

Read saved inputs only. A case is an anchor and two partners, not an independent
biological replicate. No templates, notebooks, bootstraps, or embeddings rerun.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd


def distance_label(value, limits):
    return 'near' if value <= limits[0] else 'far' if value >= limits[1] else 'middle'


def run_exploration(repo, out, caliper=.10, tail=.25):
    if tail not in [.20, .25, .30] or not 0 < caliper <= .15:
        raise ValueError('Use tail 0.20/0.25/0.30 and a chemical caliper in (0, 0.15].')
    repo, out = Path(repo).resolve(), Path(out).resolve()
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    neural = repo / 'reports/exploration_response_profiles_individual_snr_20261002/tables'
    chemistry = repo / 'reports/population_first_20260930/tables'
    old = repo / 'reports/exploration_pair_quadrants_20261002/tables'
    paths = [neural/'strain_coefficients.csv', neural/'condition_metrics.csv',
             chemistry/'aligned_chemical_report_values_all.csv',
             chemistry/'aligned_chemical_metadata.csv',
             chemistry/'aligned_chemical_reference_groups_paired.csv',
             chemistry/'aligned_taxonomy_paired.csv',
             chemistry/'aligned_chemical_log2fc_paired.csv', old/'pair_catalogue.csv']
    protected = paths + list((repo/'notebook').glob('*.ipynb'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    coefficients = pd.read_csv(paths[0], index_col=0)
    ids, cells = coefficients.index, coefficients.columns
    metrics = pd.read_csv(paths[1])
    raw = pd.read_csv(paths[2], index_col=0).loc[ids]
    metadata = pd.read_csv(paths[3], index_col=0)
    features = metadata.index[metadata.previous_complete_162_eligible]
    values = raw.loc[:, features].to_numpy(float)
    assert values.shape == (106, 162) and np.isfinite(values).all() and (values > 0).all()
    assert (metadata.loc[features, 'QCRSD'] <= .30).all()
    x = np.log2(values + 1.)
    fc = pd.read_csv(paths[6], index_col=0).loc[ids, features].to_numpy(float)
    references = pd.read_csv(paths[4], index_col=0).loc[ids].reference_group
    taxonomy = pd.read_csv(paths[5], index_col=0).loc[ids]
    dates = metrics.groupby('strain').block.apply(lambda s: ';'.join(map(str, sorted(s.unique())))).loc[ids]
    # Each cell has the same finite-date support; aggregation uses equal dates.
    cell_dates = metrics[metrics.coefficient.notna()].groupby(['strain', 'cell']).block.apply(
        lambda s: ';'.join(map(str, sorted(s.unique()))))
    assert all(cell_dates.loc[(s, c)] == dates.loc[s] for s in ids for c in cells)
    y = coefficients.to_numpy(float)
    norms = np.linalg.norm(y, axis=1)
    assert np.isfinite(y).all() and (norms > 0).all()
    u = y / norms[:, None]
    nd = np.clip(1 - u @ u.T, 0., 2.)
    dx = x[:, None, :] - x[None, :, :]
    cd = np.sqrt(np.mean(dx**2, axis=2))
    cd_fc = np.sqrt(np.mean((fc[:, None, :] - fc[None, :, :])**2, axis=2))
    iu = np.triu_indices(len(ids), 1)
    neural_limits = {q: np.quantile(nd[iu], [q, 1-q]) for q in [.20, .25, .30]}
    chemical_limits = np.quantile(cd[iu], [tail, 1-tail])
    pre = metrics.groupby(['strain', 'cell']).raw_coefficient.mean().unstack('cell').loc[ids, cells].to_numpy()
    pre_u = pre / np.linalg.norm(pre, axis=1)[:, None]
    pre_d = np.clip(1-pre_u @ pre_u.T, 0., 2.)
    previous = pd.read_csv(paths[7]).set_index('pair_id')
    assert len(previous) == 5565
    sample = pd.DataFrame({'strain': ids, 'reference': references.to_numpy(),
                           'dates': dates.to_numpy(), 'genus': taxonomy.genus_clean.to_numpy(),
                           'species': taxonomy.species_clean.to_numpy(), 'coefficient_norm': norms})
    sample['n_dates'] = sample.dates.str.count(';')+1
    sample['n_zero_cells'] = (y == 0).sum(axis=1)
    sample.to_csv(tables/'sample_context.csv', index=False)
    coefficients.to_csv(tables/'neural_coefficients.csv')
    pd.DataFrame(u, index=ids, columns=cells).to_csv(tables/'neural_unit_coefficients.csv')
    pd.DataFrame(x, index=ids, columns=features).to_csv(tables/'chemical_log2_report_plus1.csv')
    metadata.loc[features].to_csv(tables/'feature_metadata.csv')
    rows, lookup = [], {}
    equality_errors = []
    for i, j in zip(*iu):
        pair_id = '__'.join(sorted([ids[i], ids[j]]))
        prev = previous.loc[pair_id]
        assert np.isclose(nd[i,j], prev.neural, atol=1e-12)
        same_ref = references.iloc[i] == references.iloc[j]
        same_dates = dates.iloc[i] == dates.iloc[j]
        same_genus = taxonomy.genus_clean.iloc[i] == taxonomy.genus_clean.iloc[j]
        delta = x[j]-x[i]
        common_shift = delta.mean()
        shape_rms = np.sqrt(np.mean((delta-common_shift)**2))
        assert np.isclose(cd[i,j]**2, common_shift**2+shape_rms**2)
        dnorm2 = (norms[i]-norms[j])**2
        direction2 = 2*norms[i]*norms[j]*nd[i,j]
        euclidean2 = np.sum((y[i]-y[j])**2)
        assert np.isclose(euclidean2, dnorm2+direction2)
        contributions = .5*(u[i]-u[j])**2
        assert np.isclose(contributions.sum(), nd[i,j], atol=1e-12)
        if same_ref:
            equality_errors.append(abs(cd[i,j]-cd_fc[i,j]))
        record = dict(pair_id=pair_id, strain_a=ids[i], strain_b=ids[j],
            reference_a=references.iloc[i], reference_b=references.iloc[j],
            dates_a=dates.iloc[i], dates_b=dates.iloc[j],
            genus_a=taxonomy.genus_clean.iloc[i], genus_b=taxonomy.genus_clean.iloc[j],
            same_reference=same_ref, same_dates=same_dates, same_genus=same_genus,
            strict_context=bool(same_ref and same_dates and same_genus),
            chemical_distance=cd[i,j], chemical_fc162_distance=cd_fc[i,j],
            chemical_old380_distance=prev.chemical,
            chemical_label=distance_label(cd[i,j],chemical_limits),
            chemical_common_log_shift=common_shift, chemical_shape_rms=shape_rms,
            neural_distance=nd[i,j], neural_label=distance_label(nd[i,j],neural_limits[tail]),
            coefficient_norm_a=norms[i], coefficient_norm_b=norms[j],
            coefficient_rms_difference=np.sqrt(euclidean2/len(cells)),
            norm_change_squared=dnorm2, direction_change_squared=direction2,
            raw_squared_difference=euclidean2,
            direction_fraction_of_raw_difference=direction2/euclidean2 if euclidean2>1e-20 else np.nan,
            dominant_cell=cells[np.argmax(contributions)],
            dominant_cell_distance_share=contributions.max()/nd[i,j] if nd[i,j]>1e-12 else np.nan,
            neural_pre_gate=pre_d[i,j], neural_unfiltered=prev.unfiltered,
            neural_raw_curves=prev.raw, neural_snr025=prev['snr_0.25'], neural_snr075=prev['snr_0.75'],
            bootstrap_valid_fraction=prev.bootstrap_valid_fraction,
            old380_category=prev.category)
        for q, limits in neural_limits.items():
            record[f'neural_label_q{int(q*100)}'] = distance_label(nd[i,j],limits)
        rows.append(record)
        lookup[frozenset([i,j])] = record
    pairs = pd.DataFrame(rows)
    pairs['category_162'] = 'C'+pairs.chemical_label+'_N'+pairs.neural_label
    pairs.to_csv(tables/'all_pair_catalogue.csv', index=False)
    strict_pairs = pairs[pairs.strict_context].copy()
    strict_pairs.to_csv(tables/'strict_context_pairs.csv', index=False)
    assert len(strict_pairs) == 100 and max(equality_errors) < 1e-10
    stratum_rows = []
    for key, group in sample.groupby(['reference','dates','genus'], sort=True):
        ss = set(group.strain)
        pp = strict_pairs[strict_pairs.strain_a.isin(ss)&strict_pairs.strain_b.isin(ss)]
        stratum_rows.append(dict(reference=key[0],dates=key[1],genus=key[2],
            n_strains=len(group), strains=';'.join(group.strain), n_pairs=len(pp),
            n_near=int(pp.neural_label.eq('near').sum()), n_middle=int(pp.neural_label.eq('middle').sum()),
            n_far=int(pp.neural_label.eq('far').sum())))
    pd.DataFrame(stratum_rows).to_csv(tables/'strict_context_coverage.csv',index=False)
    # Generate every chemically matched anchor comparison up to the sensitivity caliper.
    case_rows = []
    for a in range(len(ids)):
        peers = [j for j in range(len(ids)) if j!=a and references.iloc[j]==references.iloc[a]
                 and dates.iloc[j]==dates.iloc[a]]
        for b,c in combinations(peers,2):
            if taxonomy.genus_clean.iloc[b] != taxonomy.genus_clean.iloc[c]:
                continue
            denom = cd[a,b]+cd[a,c]
            mismatch = 2*abs(cd[a,b]-cd[a,c])/denom if denom>1e-12 else np.inf
            if mismatch > .15:
                continue
            # Ordering does not mean either arm necessarily lies in a global tail.
            b,c = sorted([b,c], key=lambda j:(nd[a,j],ids[j]))
            rb,rc = lookup[frozenset([a,b])],lookup[frozenset([a,c])]
            db,dc = x[b]-x[a],x[c]-x[a]
            delta_cos = float(db@dc/(np.linalg.norm(db)*np.linalg.norm(dc)))
            row = dict(anchor=ids[a],partner_b=ids[b],partner_c=ids[c],
                reference=references.iloc[a],dates=dates.iloc[a],
                anchor_genus=taxonomy.genus_clean.iloc[a],partner_genus=taxonomy.genus_clean.iloc[b],
                all_same_genus=bool(taxonomy.genus_clean.iloc[a]==taxonomy.genus_clean.iloc[b]),
                chemical_distance_b=cd[a,b],chemical_distance_c=cd[a,c],chemical_mismatch=mismatch,
                chemical_change_cosine=delta_cos,
                neural_distance_b=nd[a,b],neural_distance_c=nd[a,c],neural_gap=nd[a,c]-nd[a,b],
                neural_label_b=rb['neural_label'],neural_label_c=rc['neural_label'],
                chemical_label_b=rb['chemical_label'],chemical_label_c=rc['chemical_label'],
                norm_anchor=norms[a],norm_b=norms[b],norm_c=norms[c])
            for suffix, pair in [('b',rb),('c',rc)]:
                for name in ['neural_pre_gate','neural_unfiltered','neural_raw_curves','neural_snr025',
                             'neural_snr075','bootstrap_valid_fraction','dominant_cell',
                             'dominant_cell_distance_share','direction_fraction_of_raw_difference',
                             'chemical_common_log_shift','chemical_shape_rms']:
                    row[name+'_'+suffix]=pair[name]
                # Orientation for chemical common shift is explicitly anchor -> partner.
                row['chemical_common_log_shift_'+suffix]=float((db if suffix=='b' else dc).mean())
            row['neural_gap_unfiltered']=row['neural_unfiltered_c']-row['neural_unfiltered_b']
            row['neural_gap_pre_gate']=row['neural_pre_gate_c']-row['neural_pre_gate_b']
            for q,limits in neural_limits.items():
                row[f'opposite_neural_tails_q{int(q*100)}']=bool(nd[a,b]<=limits[0] and nd[a,c]>=limits[1])
            case_rows.append(row)
    all_cases=pd.DataFrame(case_rows)
    all_cases.to_csv(tables/'candidate_matches_up_to_15pct.csv',index=False)
    main=all_cases[all_cases.all_same_genus & all_cases.chemical_mismatch.le(caliper)].copy()
    main=main.sort_values(['reference','dates','anchor_genus','anchor','partner_b','partner_c']).reset_index(drop=True)
    main.insert(0,'case_id',[f'M{i:03d}' for i in range(1,len(main)+1)])
    main.insert(1,'pdf_page',np.arange(len(main))+4)
    main.to_csv(tables/'matched_cases.csv',index=False)
    expanded=all_cases[all_cases.chemical_mismatch.le(caliper) & ~all_cases.all_same_genus]
    expanded.to_csv(tables/'excluded_cross_genus_anchor_cases.csv',index=False)
    sensitivity=[]
    for scope, mask in [('all_three_same_genus',all_cases.all_same_genus),
                        ('partners_same_genus_only',pd.Series(True,index=all_cases.index))]:
        for tol in [.05,.10,.15]:
            included=all_cases[mask & all_cases.chemical_mismatch.le(tol)]
            for q in neural_limits:
                extreme=included[included[f'opposite_neural_tails_q{int(q*100)}']]
                sensitivity.append(dict(scope=scope,caliper=tol,tail_fraction=q,
                    n_all_continuous_matches=len(included),n_opposite_tail_matches=len(extreme),
                    n_anchors=included.anchor.nunique()))
    pd.DataFrame(sensitivity).to_csv(tables/'matching_sensitivity.csv',index=False)
    # Also test pair controls without requiring a shared anchor.
    edge_counts=[]
    for q, limits in neural_limits.items():
        for tol in [.05,.10,.15]:
            count=0
            for _, group in strict_pairs.groupby(['reference_a','dates_a','genus_a']):
                low=group[group.neural_distance.le(limits[0])]
                high=group[group.neural_distance.ge(limits[1])]
                for p in low.itertuples():
                    for r in high.itertuples():
                        den=p.chemical_distance+r.chemical_distance
                        if den>1e-12 and 2*abs(p.chemical_distance-r.chemical_distance)/den<=tol:count+=1
            edge_counts.append(dict(tail_fraction=q,caliper=tol,n_opposite_tail_pair_controls=count))
    pd.DataFrame(edge_counts).to_csv(tables/'strict_opposite_tail_pair_controls.csv',index=False)
    # Signed paired changes: every case uses the same 162 measured coordinates.
    feature_rows=[]
    for row in main.itertuples():
        ia,ib,ic=[ids.get_loc(s) for s in [row.anchor,row.partner_b,row.partner_c]]
        db,dc=x[ib]-x[ia],x[ic]-x[ia]
        importance=np.sqrt((db**2+dc**2)/2)
        order=np.lexsort((np.asarray(features),-importance))
        ranks=np.empty(len(features),int);ranks[order]=np.arange(1,len(features)+1)
        for f,bb,cc,score,rank in zip(features,db,dc,importance,ranks):
            feature_rows.append(dict(case_id=row.case_id,feature=f,delta_b=bb,delta_c=cc,
                                     joint_change_rms=score,display_rank=rank))
    pd.DataFrame(feature_rows).to_csv(tables/'case_chemical_changes.csv',index=False)
    strict_pairs[strict_pairs.neural_label.eq('far')].to_csv(tables/'unmatched_far_pairs.csv',index=False)
    observations=dict(n_samples=106,n_features=162,n_cells=13,n_pairs=5565,
        n_same_reference_and_dates_pairs=int((pairs.same_reference & pairs.same_dates).sum()),
        n_strict_context_pairs=len(strict_pairs),strict_pair_neural_labels=strict_pairs.neural_label.value_counts().to_dict(),
        n_matched_cases=len(main),n_anchors=main.anchor.nunique(),
        n_matched_strains=len(set(main.anchor)|set(main.partner_b)|set(main.partner_c)),
        n_matched_contexts=len(main[['reference','dates','anchor_genus']].drop_duplicates()),
        n_opposite_tail_matches=int(main[f'opposite_neural_tails_q{int(tail*100)}'].sum()),
        n_all_three_same_species=sum(taxonomy.species_clean.loc[r.anchor]==taxonomy.species_clean.loc[r.partner_b]
                                    ==taxonomy.species_clean.loc[r.partner_c] for r in main.itertuples()),
        neural_gap_quantiles={str(q):float(main.neural_gap.quantile(q)) for q in [0,.25,.5,.75,1]},
        unfiltered_gap_same_direction_count=int(main.neural_gap_unfiltered.gt(0).sum()),
        matched_label_combinations=main.groupby(['neural_label_b','neural_label_c']).size().to_dict())
    observations['matched_label_combinations']={'/'.join(k):int(v) for k,v in observations['matched_label_combinations'].items()}
    (out/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    parameters=dict(date='2026-10-03',tail_fraction=tail,caliper=caliper,
        neural_thresholds={str(k):v.tolist() for k,v in neural_limits.items()},
        chemical_thresholds=chemical_limits.tolist(),
        primary_scope='All three strains have identical reference, full date-support set, and genus',
        matching='Shared anchor A; distinct B,C; symmetric relative chemical distance mismatch <=0.10',
        caliper_formula='2*abs(dAB-dAC)/(dAB+dAC); exclude zero denominator',
        chemical_distance='RMS log2(original reported value + 1) difference across fixed complete162/QCRSD<=0.30 panel; no imputation',
        neural_distance='1-cosine of saved 13-dimensional signed template coefficients; SNR>=0.5',
        case_order='Reference, full date support, genus, anchor, B, C; every qualifying main case displayed',
        partner_order='B has the lower primary neural distance to anchor; C has the higher; this is ordering, not an extreme-tail label',
        chemical_display='All162 signed anchor-to-partner changes; top18 by sqrt((deltaB^2+deltaC^2)/2), same features for both arms',
        sensitivity='Calipers 0.05/0.10/0.15; global neural tails 0.20/0.25/0.30; existing unfiltered, pre-gate, SNR0.25/0.75 and raw-curve distances',
        norm_decomposition='||a-b||^2=(||a||-||b||)^2+2||a||||b||(1-cosine)',
        chemical_level_decomposition='RMS(delta)^2=mean(delta)^2+RMS(delta-mean(delta))^2; mean log shift is not total concentration',
        reference='Numerical FC denominator group; specific experimental identity not established by this analysis',
        limitations=['Distance length matching does not match chemical direction or full composition',
                     'Same genus need not mean same species; same date need not remove stimulus-order effects',
                     'Chemical and neural measurements are not from the same actual stimulus aliquot',
                     'Screened zeros are analysis choices; original source missingness remains outside the 162-feature scope',
                     'Cases share strains and are not independent replicates; no p-values, causal, equivalence, or performance claim'],
        source_sha256=hashes)
    (out/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    verification=dict(inputs_and_notebooks_unchanged=unchanged,
        n_missing_primary_chemical_values=int(np.isnan(values).sum()),
        n_neural_zero_vectors=int((norms==0).sum()),
        same_reference_raw_vs_fc162_max_distance_error=max(equality_errors),
        neural_distances_match_previous=True,raw_norm_direction_decomposition_passed=True,
        chemical_level_shape_decomposition_passed=True,
        all_main_cases_strictly_matched=bool(main.all_same_genus.all() and main.chemical_mismatch.le(caliper).all()),
        n_case_feature_rows=len(feature_rows),n_main_cases=len(main),
        no_full_notebook_execution=True,no_new_templates_or_bootstrap=True)
    assert unchanged and len(feature_rows)==len(main)*len(features)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(observations,indent=2))
    return main


if __name__ == '__main__':
    output=Path(__file__).resolve().parents[1]
    run_exploration(output.parents[1],output)

"""Chemical-only group construction; no neural responses are read.

Run from any directory with the project .pixi/envs/default/bin/python.
Outputs are restricted to tables/chemical_groups_* and logs/chemical_groups_*.
One row is one strain-linked LC-MS reference profile, not a culture replicate
or verified neural stimulus concentration. Names/classes remain annotations.

Primary scores: median of member log2(report+1) values standardized with each
feature's mean and sample SD over the 106 reference strains. Fixed annotation
membership is retained, including incongruent members in the audit tables.
Chemical-only coherence rules identify a short usable annotation panel. These
rules and the data modules are exploratory selection, not prespecified biology.
For prospective CV, refit scaling on training strains; group selection/module
construction based on all 106 strains is transductive and must be disclosed or
rerun in training folds. No inference or biological mechanism is claimed here.
"""
from pathlib import Path
import hashlib
import json
import platform
import unicodedata

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT / 'reports/exploration_20260929/tables'
TABLES = OUT / 'tables'
LOGS = OUT / 'logs'
SEED = 2026093004
N_BOOT = 200
MODULE_MIN_RHO = .65
LEVELS = ['SuperClass', 'Class', 'SubClass', 'DirectParent']
MIN_GROUP_N = 3
MAX_CANDIDATE_N = 12
MIN_PAIR_MEDIAN = .20
MIN_POSITIVE_FRACTION = .85
MIN_MEMBER_REST_RHO = .10


def clean_label(value):
    if pd.isna(value):
        return None
    value = unicodedata.normalize('NFKC', str(value)).strip()
    return None if value.lower() in {'', 'na', 'nan', 'none'} else value


def rho(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def pair_values(x):
    c = x.corr(method='spearman').to_numpy()
    return c[np.triu_indices(c.shape[0], 1)]


def cluster_labels(x):
    c = x.corr(method='spearman').fillna(0).to_numpy()
    distance = np.clip(1 - c, 0, 2)
    np.fill_diagonal(distance, 0)
    return fcluster(linkage(squareform(distance, checks=False), method='complete'),
                    t=1 - MODULE_MIN_RHO, criterion='distance')


def residual(y, z):
    """Ordinary projection for strain-level chemical descriptors; no p values."""
    z = np.asarray(z, float)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    keep = s > np.finfo(float).eps * max(z.shape) * s[0]
    q = u[:, keep]
    return np.asarray(y) - q @ (q.T @ np.asarray(y)), int(keep.sum())


def category_diagnostics(score, category):
    centered = score - score.groupby(category).transform('mean')
    total_ss = float(np.sum((score - score.mean()) ** 2))
    within_ss = float(np.sum(centered ** 2))
    n, k = len(score), category.nunique()
    r2 = 1 - within_ss / total_ss
    adj = 1 - (within_ss / (n - k)) / (total_ss / (n - 1)) if n > k else np.nan
    return r2, adj, np.sqrt(within_ss / total_ss)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    inputs = ['chemical_log.csv', 'chemical_raw.csv', 'chemical_feature_metadata.csv',
              'chemical_reference_groups.csv', 'taxonomy.csv']
    manifest = {f: hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in inputs}
    meta = pd.read_csv(SOURCE/'chemical_feature_metadata.csv', index_col=0)
    raw = pd.read_csv(SOURCE/'chemical_raw.csv', index_col=0)
    log = pd.read_csv(SOURCE/'chemical_log.csv', index_col=0)
    tax = pd.read_csv(SOURCE/'taxonomy.csv', index_col=0)
    reference = pd.read_csv(SOURCE/'chemical_reference_groups.csv', index_col=0)
    assert all(df.index.is_unique for df in [meta, raw, log, tax, reference])
    assert set(raw.index) == set(log.index) == set(tax.index) == set(reference.index)
    ids = sorted(log.index)
    complete = (meta.QCRSD <= .30) & raw.notna().all(axis=0).reindex(meta.index)
    assert complete.equals(meta.complete_eligible)
    names = sorted(meta.index[complete])
    assert len(names) == 162 and len(ids) == 106
    x = log.loc[ids, names]
    assert np.isfinite(x.to_numpy()).all()
    assert np.allclose(x, np.log2(raw.loc[ids, names] + 1), rtol=1e-12, atol=1e-12)
    m = meta.loc[names].copy()
    for level in LEVELS:
        m[level] = m[level].map(clean_label)
    taxonomy = tax.loc[ids, ['genus_clean', 'species_clean']].apply(lambda c: c.map(clean_label))
    assert taxonomy.notna().all().all()
    references = reference.loc[ids, 'reference_group']
    feature_mean = x.mean()
    feature_sd = x.std(ddof=1)
    assert feature_sd.gt(0).all()
    z = (x - feature_mean) / feature_sd
    scaling = pd.DataFrame({'mean_log2_report_plus1':feature_mean, 'sd_log2_report_plus1':feature_sd,
                            'n_strains':len(x), 'QCRSD':m.QCRSD})
    scaling.index.name = 'feature'
    scaling.to_csv(TABLES/'chemical_groups_scaling.csv')
    total_score = z.median(axis=1)
    total_log = x.median(axis=1)
    genus_sizes = taxonomy.genus_clean.value_counts()
    within_mask = taxonomy.genus_clean.map(genus_sizes).ge(3)
    within_x = x.loc[within_mask] - x.loc[within_mask].groupby(taxonomy.loc[within_mask, 'genus_clean']).transform('mean')

    # Deduplicate identical memberships, preferring the most specific level.
    # Annotation aliases remain in the group table; NA is never a category.
    definitions = {}
    for level in reversed(LEVELS):
        for label, rows in m.groupby(level):
            if len(rows) < MIN_GROUP_N:
                continue
            key = tuple(sorted(rows.index))
            alias = {'level':level, 'label':label}
            if key not in definitions:
                definitions[key] = {'kind':'annotation', 'level':level, 'label':label, 'aliases':[]}
            definitions[key]['aliases'].append(alias)
    groups = []
    for i, (members, info) in enumerate(sorted(definitions.items(), key=lambda item:(item[1]['level'], item[1]['label'])), 1):
        groups.append({'group_id':f'ann_{i:02d}', 'members':list(members), **info})

    # Positive complete-linkage blocks: every pair in a module has rho >= .65
    # in the original reference panel. No pathway label or neural target is used.
    labels = cluster_labels(x)
    modules = [list(x.columns[labels == k]) for k in sorted(set(labels)) if np.sum(labels == k) >= MIN_GROUP_N]
    modules.sort(key=lambda members:(-len(members), tuple(members)))
    for i, members in enumerate(modules, 1):
        groups.append({'group_id':f'module_{i:02d}', 'members':members, 'kind':'data_module',
                       'level':'positive_covariance', 'label':f'Positive covariance module {i:02d}', 'aliases':[]})

    all_scores, collapsed_scores, audit, members_out, genus_out, pairs_out = {}, {}, [], [], [], []
    for group in groups:
        gid, members = group['group_id'], group['members']
        score = z[members].median(axis=1)
        all_scores[gid] = score
        # Conservative redundancy check: equal weight per reported Mass/column
        # family instead of per annotation. Shared transitions need not denote
        # identical molecules, so this is a sensitivity score, not peak merging.
        families = {}
        for feature in members:
            mass = clean_label(m.loc[feature,'Mass'])
            key = (mass, clean_label(m.loc[feature,'column'])) if mass is not None else (feature, None)
            families.setdefault(key, []).append(feature)
        collapsed = pd.DataFrame({str(i):z[ff].median(axis=1) for i,ff in enumerate(families.values())}).median(axis=1)
        collapsed_scores[gid] = collapsed
        values = pair_values(x[members])
        within_pairs = pair_values(within_x[members])
        member_rest = {f:rho(z[f], z[[g for g in members if g != f]].median(axis=1)) for f in members}
        drop_corr = [rho(score, z[[g for g in members if g != f]].median(axis=1)) for f in members]
        positive_fraction = float(np.mean(values > 0))
        coherence = (MIN_GROUP_N <= len(members) <= MAX_CANDIDATE_N and
                     np.median(values) >= MIN_PAIR_MEDIAN and
                     positive_fraction >= MIN_POSITIVE_FRACTION and
                     min(member_rest.values()) >= MIN_MEMBER_REST_RHO)
        candidate = group['kind'] == 'annotation' and coherence
        leave_group_out_total = z.drop(columns=members).median(axis=1)
        r2g, adjg, residual_fraction = category_diagnostics(score, taxonomy.genus_clean)
        r2r, adjr, _ = category_diagnostics(score, references)
        design = pd.concat([pd.get_dummies(taxonomy.genus_clean, dtype=float),
                            pd.get_dummies(references, dtype=float),
                            leave_group_out_total.rename('leave_group_out_panel_level')], axis=1)
        res, rank = residual(score, design)
        joint_r2 = 1 - np.sum(res ** 2) / np.sum((score - score.mean()) ** 2)
        partial_features, _ = residual(x[members], design)
        adjusted_pairs = pair_values(pd.DataFrame(partial_features, columns=members))
        score_genus = score - score.groupby(taxonomy.genus_clean).transform('mean')
        total_genus = leave_group_out_total - leave_group_out_total.groupby(taxonomy.genus_clean).transform('mean')
        row = dict(group_id=gid, kind=group['kind'], annotation_level=group['level'], label=group['label'],
                   aliases=json.dumps(group['aliases'], ensure_ascii=False), n_members=len(members),
                   primary_candidate=bool(candidate), exploratory_large_module=group['kind']=='data_module' and len(members)>=8,
                   n_strains=len(ids), n_genera=taxonomy.genus_clean.nunique(), n_reference_groups=references.nunique(),
                   n_within_genus_rows=int(within_mask.sum()), n_genera_with_at_least_3=int(genus_sizes.ge(3).sum()),
                   pair_rho_min=float(values.min()), pair_rho_median=float(np.median(values)),
                   pair_positive_fraction=positive_fraction, member_rest_rho_min=min(member_rest.values()),
                   within_genus_pair_rho_median=float(np.median(within_pairs)),
                   joint_adjusted_pair_rho_median=float(np.median(adjusted_pairs)),
                   score_mean_sensitivity_rho=rho(score, z[members].mean(axis=1)),
                   leave_one_member_score_rho_min=min(drop_corr),
                   n_mass_column_families=len(families), transition_balanced_score_rho=rho(score,collapsed),
                   panel_level_rho=rho(score,total_score), raw_median_log_level_rho=rho(score,total_log),
                   leave_group_out_panel_level_rho=rho(score,leave_group_out_total),
                   within_genus_panel_level_rho=rho(score_genus.loc[within_mask],total_genus.loc[within_mask]),
                   genus_R2=r2g, genus_adjusted_R2=adjg, within_genus_sd_fraction=residual_fraction,
                   reference_R2=r2r, reference_adjusted_R2=adjr,
                   genus_reference_panel_joint_R2=float(joint_r2), joint_nuisance_rank=rank,
                   residual_after_genus_reference_panel_sd_fraction=float(np.sqrt(max(0,1-joint_r2))),
                   qc_rsd_min=float(m.loc[members,'QCRSD'].min()), qc_rsd_max=float(m.loc[members,'QCRSD'].max()),
                   n_pairs_above_rho_095=int(np.sum(values > .95)),
                   warning='Very correlated members may be redundant annotations or analytical signals; no molecular identity validation.' if np.any(values>.95) else '')
        audit.append(row)
        for f in members:
            members_out.append(dict(group_id=gid, kind=group['kind'], primary_candidate=bool(candidate), feature=f,
                                    member_vs_other_members_rho=member_rest[f],
                                    **m.loc[f,['QCRSD','qc_n_detected','Class','SubClass','DirectParent','SuperClass','Mass','RT','column']].to_dict()))
        for genus, q in score.groupby(taxonomy.genus_clean):
            if len(q) >= 3:
                genus_out.append(dict(group_id=gid, genus=genus, n_strains=len(q), mean=q.mean(), sd=q.std(ddof=1),
                                      q10=q.quantile(.1), q90=q.quantile(.9), minimum=q.min(), maximum=q.max(),
                                      min_strain=q.idxmin(), max_strain=q.idxmax()))
        corr = x[members].corr(method='spearman')
        for j, f in enumerate(members):
            for g in members[j+1:]:
                pairs_out.append(dict(group_id=gid, feature1=f, feature2=g, rho=float(corr.loc[f,g])))

    # Internal module stability: resample the reference strains, then reconstruct
    # all clusters. This is neither technical repeatability nor new-sample validation.
    rng = np.random.default_rng(SEED)
    coassignment = np.zeros((len(names), len(names)), float)
    for _ in range(N_BOOT):
        boot_labels = cluster_labels(x.iloc[rng.integers(0, len(x), len(x))].reset_index(drop=True))
        coassignment += boot_labels[:,None] == boot_labels[None,:]
    coassignment /= N_BOOT
    stability = []
    for group in groups:
        if group['kind'] != 'data_module':
            continue
        ix = [names.index(f) for f in group['members']]
        vals = coassignment[np.ix_(ix,ix)][np.triu_indices(len(ix),1)]
        stability.append(dict(group_id=group['group_id'], n_members=len(ix), bootstrap_draws=N_BOOT,
                              pair_coassignment_min=float(vals.min()), pair_coassignment_median=float(np.median(vals)),
                              pair_coassignment_p10=float(np.quantile(vals,.1)), pair_coassignment_max=float(vals.max())))
    stability = pd.DataFrame(stability)
    audit = pd.DataFrame(audit).merge(stability,on=['group_id','n_members'],how='left',validate='one_to_one')
    scores = pd.DataFrame(all_scores, index=ids)
    scores.index.name = 'sample_id'
    candidates = audit.loc[audit.primary_candidate,'group_id'].tolist()
    optional_modules = audit.loc[audit.exploratory_large_module,'group_id'].tolist()
    assert len(candidates) <= 10 and all(scores[candidates].notna().all())
    assert all(audit.loc[audit.kind.eq('data_module'),'pair_rho_min'] >= MODULE_MIN_RHO-1e-12)
    scores[candidates].to_csv(TABLES/'chemical_groups_scores.csv')
    scores.to_csv(TABLES/'chemical_groups_all_scores.csv')
    scores[optional_modules].to_csv(TABLES/'chemical_groups_exploratory_module_scores.csv')
    descriptors = taxonomy.copy()
    descriptors['reference_group'] = references
    descriptors['panel_level_median_standardized_log'] = total_score
    descriptors['panel_median_log2_report_plus1'] = total_log
    descriptors.to_csv(TABLES/'chemical_groups_strain_metadata.csv',index_label='sample_id')
    audit.to_csv(TABLES/'chemical_groups_audit.csv',index=False)
    pd.DataFrame(members_out).to_csv(TABLES/'chemical_groups_members.csv',index=False)
    pd.DataFrame(genus_out).to_csv(TABLES/'chemical_groups_within_genus.csv',index=False)
    pd.DataFrame(pairs_out).to_csv(TABLES/'chemical_groups_pair_correlations.csv',index=False)
    stability.to_csv(TABLES/'chemical_groups_module_stability.csv',index=False)
    scores[candidates+optional_modules].corr(method='spearman').to_csv(TABLES/'chemical_groups_score_correlations.csv')
    pd.DataFrame(collapsed_scores,index=ids)[candidates+optional_modules].to_csv(
        TABLES/'chemical_groups_transition_balanced_scores.csv',index_label='sample_id')
    # Source files are hashed again so concurrent reads cannot conceal mutations.
    assert manifest == {f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in inputs}
    params = dict(seed=SEED, module_bootstrap_draws=N_BOOT, module_min_positive_pair_rho=MODULE_MIN_RHO,
                  hierarchy_linkage='complete', annotation_min_n=MIN_GROUP_N, annotation_max_candidate_n=MAX_CANDIDATE_N,
                  annotation_min_pair_median=MIN_PAIR_MEDIAN, annotation_min_positive_pair_fraction=MIN_POSITIVE_FRACTION,
                  annotation_min_member_rest_rho=MIN_MEMBER_REST_RHO,
                  score_formula='median over members of [(log2(report_ng_mL+1)-training_feature_mean)/training_feature_SD_ddof1]',
                  saved_scores_scaling_scope='all 106 strains; descriptive/transductive; refit mean/SD for predictive CV',
                  current_annotation_selection_scope='chemical-only coherence over all 106 strains; no neural input',
                  total_index='median standardized complete-QC panel, not total concentration; leave-group-out version used in adjustment',
                  metadata_levels=LEVELS, fixed_panel_members=names)
    summary = dict(status='passed', n_strains=len(ids), n_features=len(names),
                   n_annotation_groups=int(audit.kind.eq('annotation').sum()), n_primary_candidates=len(candidates),
                   primary_candidates=audit.loc[audit.primary_candidate,['group_id','label','n_members']].to_dict('records'),
                   n_covariance_modules=len(stability), exploratory_large_modules=optional_modules,
                   neural_data_read=False, input_sha256=manifest, parameters=params,
                   python=platform.python_version(), numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                   caveats=['Annotation is not confirmed molecular identity, pathway membership, or biological activity.',
                            'One reference chemical profile per strain; no verified culture/exposure match or independent chemical biological replication.',
                            'Class aggregation is chemically screened, not prespecified; full-panel modules/scaling are transductive.',
                            'Bootstrap resamples strains, not culture or measurement replicates, and does not remove taxonomy/reference confounding.',
                            'Transition-balanced scores are a conservative annotation-redundancy sensitivity, not validated molecular deduplication.',
                            'R2 values describe categorical separation in this panel; they do not identify causal genus, batch, or reference effects.',
                            'Unadjusted/adjusted scores and pair correlations are descriptive; no p values or mechanism claims.'])
    (LOGS/'chemical_groups_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:summary[k] for k in ['status','n_strains','n_features','n_annotation_groups','n_primary_candidates','primary_candidates','n_covariance_modules','exploratory_large_modules']},indent=2))
    print(audit.loc[audit.primary_candidate | audit.exploratory_large_module,
          ['group_id','label','n_members','pair_rho_median','member_rest_rho_min','within_genus_pair_rho_median',
           'leave_group_out_panel_level_rho','genus_R2','within_genus_sd_fraction','reference_R2',
           'residual_after_genus_reference_panel_sd_fraction','pair_coassignment_median','pair_coassignment_min']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()

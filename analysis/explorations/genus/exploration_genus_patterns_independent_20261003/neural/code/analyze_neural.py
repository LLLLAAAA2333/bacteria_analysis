"""Step 4 only. Run with run_analysis(repo_root), including from an existing Notebook.

Inputs: saved strain x 13 signed unit coefficients (gated and same-template
pre-gate), genus/date metadata, and optional neural support audit. No raw traces,
chemical matrix, score, or step-5 comparison is used. Output arrays and csv files
retain all input neuron coordinates. No packages are installed.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import scipy

REL = Path('reports/exploration_genus_patterns_independent_20261003/neural')
INPUT_REL = Path('reports/exploration_chemical_pattern_direct_report_20261003/tables')
EPS = 1e-12
SMALL_OFFSET = 0.05


def cosine(a, b):
    """Cosine of two coordinate vectors; zero length is explicitly missing."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (norm_a * norm_b)) if min(norm_a, norm_b) > EPS else np.nan


def profile_summary(x, genus):
    """Return G x K unnormalized means, K-vector equal-genus ref, centered means."""
    means = x.groupby(genus).mean().sort_index()
    reference = means.mean(axis=0)
    return means, reference, means.subtract(reference, axis=1)


def neural_order(profiles):
    """All coordinates, Euclidean / average linkage; tree is only a display order."""
    if len(profiles) <= 2:
        return profiles.index.tolist()
    tree = linkage(pdist(profiles.to_numpy(), metric='euclidean'),
                   method='average', optimal_ordering=True)
    return profiles.index[leaves_list(tree)].tolist()


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(directory / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def heatmap(ax, values, labels, neurons, limit):
    im = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit,
                   interpolation='nearest')
    ax.set_xticks(np.arange(len(neurons)), neurons, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_results(out, x, genus, means, ref, centered, order, holdouts, pre_centered):
    """English-only complete combination figures, symmetric scales, no row z score."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none'})
    neurons = x.columns.tolist()
    counts = genus.value_counts()
    labels = [f'{g}  (n={counts[g]})' for g in order]
    c = centered.loc[order]
    center_limit = np.ceil(np.abs(c.to_numpy()).max() * 20) / 20
    fig, ax = plt.subplots(figsize=(10.9, 6.7), constrained_layout=True)
    im = heatmap(ax, c.to_numpy(), labels, neurons, center_limit)
    ax.set_title('Genus-associated neural combinations', loc='left', pad=16,
                 fontsize=14, fontweight='bold')
    bar = fig.colorbar(im, ax=ax, pad=0.03, shrink=.8)
    bar.set_label('Mean unit coefficient − equal-genus reference')
    save_figure(fig, out/'figures', '01_neural_genus_centered_profiles')

    strain_order = []
    for g in order:
        group = x.loc[genus.eq(g)].sort_index()
        strain_order.extend(neural_order(group))
    individual = x.loc[strain_order].subtract(ref, axis=1)
    individual_limit = np.ceil(np.abs(individual.to_numpy()).max() * 10) / 10
    fig, ax = plt.subplots(figsize=(11.2, 18.8), constrained_layout=True)
    im = heatmap(ax, individual.to_numpy(), strain_order, neurons, individual_limit)
    ax.tick_params(axis='y', labelsize=6.6)
    start = 0
    for g in order:
        n = int(counts[g]); mid = start + (n-1)/2
        ax.text(-2.0, mid, f'{g} (n={n})', ha='right', va='center', fontsize=9,
                transform=ax.transData)
        if start:
            ax.axhline(start-.5, color='white', lw=1.5)
        start += n
    ax.set_title('Every strain, grouped by genus', loc='left', pad=16,
                 fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, pad=.025, shrink=.35).set_label(
        'Strain unit coefficient − equal-genus reference')
    save_figure(fig, out/'figures', '02_neural_all_strains_centered_profiles')

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.8), sharey=True,
                             gridspec_kw={'width_ratios': [1.25, 1]},
                             constrained_layout=True)
    for i, g in enumerate(order):
        h = holdouts[holdouts.genus.eq(g)].sort_values('strain')
        offsets = np.linspace(-.17, .17, len(h)) if len(h)>1 else np.array([0])
        axes[0].scatter(h.heldout_direction_cosine, i+offsets, color='#366b96',
                        alpha=.75, s=18)
        axes[1].scatter(h.center_direction_cosine, i+offsets, color='#718a58',
                        alpha=.75, s=18)
    axes[0].axvline(0, color='.5', ls='--', lw=.8)
    axes[0].set_xlim(-1.05, 1.05)
    axes[1].set_xlim(-1.05, 1.05)
    axes[0].set_yticks(range(len(order)), labels)
    axes[0].invert_yaxis()
    axes[0].set_title('Held-out strain vs remaining genus', loc='left', pad=13)
    axes[1].set_title('Remaining genus vs full genus center', loc='left', pad=13)
    for ax in axes:
        ax.set_xlabel('Cosine of the centered 13-neuron combination')
        ax.grid(axis='x', alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    save_figure(fig, out/'figures', '03_neural_leave_one_strain_out')

    both_limit = np.ceil(max(np.abs(c.to_numpy()).max(),
                            np.abs(pre_centered.to_numpy()).max())*20)/20
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.7), sharey=True,
                             constrained_layout=True)
    im = heatmap(axes[0], c.to_numpy(), labels, neurons, both_limit)
    heatmap(axes[1], pre_centered.loc[order].to_numpy(), labels, neurons, both_limit)
    axes[1].tick_params(axis='y', labelleft=False)
    axes[0].set_title('SNR-gated unit profiles', loc='left', pad=14)
    axes[1].set_title('Same-template pre-gate unit profiles', loc='left', pad=14)
    fig.colorbar(im, ax=axes, pad=.02, shrink=.8).set_label(
        'Mean unit coefficient − equal-genus reference')
    save_figure(fig, out/'figures', '04_neural_pre_gate_sensitivity')
    return {'genus_order': order, 'neuron_order': neurons, 'strain_order': strain_order,
            'main_center_color_limit': float(center_limit),
            'individual_color_limit': float(individual_limit),
            'sensitivity_color_limit': float(both_limit)}


def run_analysis(repo_root, out=None):
    """Compute once into a fresh output directory; saved scientific results are immutable.

    Parameters
    ----------
    repo_root : path
        Repository holding the fixed input CSVs.
    out : path, optional
        Fresh output directory. Relative paths resolve beneath repo_root.
        Default is the original report location, which refuses a second run.
    """
    root = Path(repo_root).resolve()
    out = root/REL if out is None else Path(out)
    out = (root/out).resolve() if not out.is_absolute() else out.resolve()
    has_tables = (out/'tables').is_dir() and any((out/'tables').iterdir())
    has_core = any((out/name).exists() for name in ['parameters.json', 'source_manifest.json'])
    if has_tables or has_core:
        raise FileExistsError(f'Refusing to overwrite saved scientific results: {out}. Choose a fresh out directory.')
    for d in ['tables','figures','verification']:
        (out/d).mkdir(parents=True, exist_ok=True)
    paths = {
        'neural_unit': root/INPUT_REL/'neural_unit_coefficients.csv',
        'neural_pre_gate_unit': root/INPUT_REL/'neural_pre_gate_unit_coefficients.csv',
        'sample_context': root/INPUT_REL/'sample_context.csv',
        'strain_audit': root/'reports/exploration_response_profiles_individual_snr_20261002/tables/strain_audit.csv',
    }
    sources = {key:{'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bytes':path.stat().st_size} for key,path in paths.items()}
    all_x = pd.read_csv(paths['neural_unit'], index_col='strain')
    all_pre = pd.read_csv(paths['neural_pre_gate_unit'], index_col='strain')
    context = pd.read_csv(paths['sample_context'], dtype={'strain':str, 'dates':str}).set_index('strain')
    assert all_x.index.is_unique and all_pre.index.is_unique and context.index.is_unique
    assert all_x.index.equals(all_pre.index) and all_x.columns.equals(all_pre.columns)
    assert set(all_x.index)==set(context.index)
    assert all_x.shape==(106,13) and np.isfinite(all_x.to_numpy()).all()
    assert np.isfinite(all_pre.to_numpy()).all()
    assert np.allclose(np.linalg.norm(all_x,axis=1),1,atol=1e-10)
    assert np.allclose(np.linalg.norm(all_pre,axis=1),1,atol=1e-10)
    context=context.loc[all_x.index].copy()
    all_counts=context.genus.value_counts()
    genera=sorted(all_counts[all_counts.ge(2)].index)
    main=context.genus.isin(genera)
    x=all_x.loc[main].copy(); pre=all_pre.loc[main].copy()
    meta=context.loc[main].copy(); genus=meta.genus
    assert x.shape==(90,13) and len(genera)==13
    means,ref,centered=profile_summary(x,genus)
    pre_means,pre_ref,pre_centered=profile_summary(pre,genus)
    order=neural_order(centered)
    individual=x.subtract(ref,axis=1)
    pre_individual=pre.subtract(pre_ref,axis=1)
    audit=pd.read_csv(paths['strain_audit'])
    audit=audit[audit.strain.isin(x.index)].copy()
    audit.to_csv(out/'tables/neural_qc_audit.csv',index=False)
    meta['genus_n']=genus.map(genus.value_counts())
    meta['main_analysis']=True
    meta.to_csv(out/'tables/main_membership.csv')
    singleton=context.loc[~main].copy();singleton['main_analysis']=False
    singleton.to_csv(out/'tables/singleton_coverage_only.csv')
    pd.DataFrame({'reference':ref,'pre_gate_reference':pre_ref}).to_csv(out/'tables/equal_genus_reference.csv',index_label='neuron')
    for name,data in [('genus_mean_unit_profiles',means),('genus_centered_profiles',centered),
                      ('pre_gate_genus_mean_unit_profiles',pre_means),
                      ('pre_gate_genus_centered_profiles',pre_centered),
                      ('strain_unit_profiles',x),('strain_centered_profiles',individual),
                      ('pre_gate_strain_centered_profiles',pre_individual)]:
        data.to_csv(out/'tables'/f'{name}.csv')
    rows=[]; holds=[]; summaries=[]; sensitivity=[]; neuron_sign_loo={}
    for g in genera:
        group=x.loc[genus.eq(g)]; n=len(group)
        mu=means.loc[g].to_numpy(); delta=centered.loc[g].to_numpy()
        norm=float(np.linalg.norm(delta)); dispersion=float(np.sqrt(np.mean(np.sum((group.to_numpy()-mu)**2,axis=1))))
        ah=audit[audit.strain.isin(group.index)]
        records=[]; signs=[]
        for s in group.index:
            remaining=group.drop(index=s).mean(axis=0).to_numpy()
            loo_ref=ref.to_numpy()+(remaining-mu)/len(genera)
            loo_delta=remaining-loo_ref
            heldout=x.loc[s].to_numpy()-loo_ref
            signs.append(np.sign(loo_delta)==np.sign(delta))
            r={'genus':g,'strain':s,'n_remaining':n-1,
               'center_direction_cosine':cosine(loo_delta,delta),
               'center_l2_change':float(np.linalg.norm(loo_delta-delta)),
               'center_offset_norm':float(np.linalg.norm(loo_delta)),
               'heldout_direction_cosine':cosine(heldout,loo_delta),
               'heldout_offset_norm':float(np.linalg.norm(heldout)),
               'center_coordinate_sign_fraction':float(np.mean(signs[-1])),
               'small_offset_direction_caution':bool(norm<SMALL_OFFSET or np.linalg.norm(loo_delta)<SMALL_OFFSET)}
            holds.append(r);records.append(r)
        gh=pd.DataFrame(records)
        neuron_sign_loo[g]=np.mean(signs,axis=0)
        for j,neuron in enumerate(x.columns):
            vals=group[neuron].to_numpy(); residual=vals-ref[neuron]
            rows.append({'genus':g,'n':n,'neuron':neuron,'mean_unit':float(vals.mean()),
                         'reference':float(ref[neuron]),'centered_mean':float(delta[j]),
                         'median_unit':float(np.median(vals)), 'sd_unit':float(np.std(vals,ddof=1)),
                         'q10_unit':float(np.quantile(vals,.1)),'q25_unit':float(np.quantile(vals,.25)),
                         'q75_unit':float(np.quantile(vals,.75)),'q90_unit':float(np.quantile(vals,.9)),
                         'same_sign_count':int(np.sum(np.sign(residual)==np.sign(delta[j]))),
                         'same_sign_fraction':float(np.mean(np.sign(residual)==np.sign(delta[j]))),
                         'zero_fraction':float(np.mean(vals==0)),
                         'loo_coordinate_sign_fraction':float(neuron_sign_loo[g][j])})
        top=np.argsort(-np.abs(delta))[:3]
        all_dates=set(d for values in meta.loc[group.index,'dates'].fillna('') for d in values.replace('|',';').split(';') if d)
        species=meta.loc[group.index,'species'].value_counts(dropna=False)
        summaries.append({'genus':g,'n':n,'mean_profile_norm':float(np.linalg.norm(mu)),
                          'centered_profile_norm':norm,'within_genus_rms_dispersion':dispersion,
                          'offset_to_within_rms_ratio':norm/dispersion if dispersion>EPS else np.nan,
                          'small_offset_direction_caution':bool(norm<SMALL_OFFSET),
                          'heldout_positive_count':int(gh.heldout_direction_cosine.gt(0).sum()),
                          'heldout_valid_count':int(gh.heldout_direction_cosine.notna().sum()),
                          'heldout_positive_fraction':float(gh.heldout_direction_cosine.gt(0).sum()/gh.heldout_direction_cosine.notna().sum()) if gh.heldout_direction_cosine.notna().any() else np.nan,
                          'heldout_cosine_min':gh.heldout_direction_cosine.min(),
                          'heldout_cosine_median':gh.heldout_direction_cosine.median(),
                          'loo_center_cosine_min':gh.center_direction_cosine.min(),
                          'loo_center_cosine_median':gh.center_direction_cosine.median(),
                          'loo_center_l2_change_max':gh.center_l2_change.max(),
                          'gated_zero_fraction':float((group==0).to_numpy().mean()),
                          'recorded_animals_min':int(ah.n_animals_recorded.min()),
                          'recorded_animals_median':float(ah.n_animals_recorded.median()),
                          'n_species_labels':int(len(species)),
                          'dominant_species_label':str(species.index[0]),
                          'dominant_species_n':int(species.iloc[0]),
                          'n_date_labels':len(all_dates),'date_labels':';'.join(sorted(all_dates)),
                          'top3_abs_coordinate_names':';'.join(x.columns[top]),
                          'top3_centered_values':';'.join(f'{v:+.6f}' for v in delta[top])})
        pre_delta=pre_centered.loc[g].to_numpy()
        sensitivity.append({'genus':g,'n':n,'centered_profile_cosine':cosine(delta,pre_delta),
                            'centered_profile_l2_change':float(np.linalg.norm(delta-pre_delta)),
                            'pre_gate_centered_norm':float(np.linalg.norm(pre_delta)),
                            'centered_coordinate_sign_fraction':float(np.mean(np.sign(delta)==np.sign(pre_delta))),
                            'strain_centered_cosine_min':min(cosine(individual.loc[s],pre_individual.loc[s]) for s in group.index),
                            'strain_centered_cosine_median':np.median([cosine(individual.loc[s],pre_individual.loc[s]) for s in group.index])})
    summary=pd.DataFrame(summaries).set_index('genus').loc[order]
    cell_summary=pd.DataFrame(rows);holdouts=pd.DataFrame(holds)
    summary.to_csv(out/'tables/genus_summary.csv')
    cell_summary.to_csv(out/'tables/genus_neuron_summary.csv',index=False)
    holdouts.to_csv(out/'tables/leave_one_strain_out.csv',index=False)
    pd.DataFrame(sensitivity).set_index('genus').loc[order].to_csv(out/'tables/pre_gate_sensitivity_summary.csv')
    contrast_rows=[]
    for i,g1 in enumerate(genera):
        for g2 in genera[i+1:]:
            diff=means.loc[g1]-means.loc[g2]
            contrast_rows.append({'genus_a':g1,'genus_b':g2,'center_distance':float(np.linalg.norm(diff)), **diff.to_dict()})
    pd.DataFrame(contrast_rows).to_csv(out/'tables/all_genus_pair_contrasts.csv',index=False)
    (pre_centered-centered).to_csv(out/'tables/pre_gate_minus_gated_centered.csv')
    display=plot_results(out,x,genus,means,ref,centered,order,holdouts,pre_centered)
    params={'analysis_date':'2026-10-03','n_main_strains':len(x),'n_main_genera':len(genera),
            'n_singletons':len(singleton),'reference':'unweighted mean of 13 unnormalized genus means',
            'genus_order_method':'average-linkage Euclidean on neural centered means; optimal leaf order; display only',
            'strain_order_method':'same method within genus; strain ID for n=2',
            'EPS':EPS,'small_offset_guard':SMALL_OFFSET,'top_coordinate_examples':3,
            'no_row_zscore':True,'no_mean_renormalization':True,'no_chemical_data':True,
            'leaveout_reference_recomputed':True,'same_template_pre_gate':True,
            **display, 'versions':{'python':platform.python_version(),'numpy':np.__version__,
                                 'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'scipy':scipy.__version__}}
    (out/'parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    (out/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    (out/'figures/CAPTIONS.md').write_text('''# Figure captions

1. **Genus-associated neural combinations.** Each row is a mean of the strain unit vectors, minus the equal-weight mean of the 13 genus means. All 13 neuron coordinates are retained. Colors show signed template-coordinate differences, not firing rate, excitation/inhibition, proportions, or response gain. Mean vectors are not renormalized and rows are not z-scored. Row order uses only neural Euclidean differences; the tree is a display aid. n is observed strains, not independent culture replicates.
2. **Every strain, grouped by genus.** All 90 unit vectors, minus the same reference as figure 1. Genus order matches figure 1; strains within a genus are ordered by neural Euclidean similarity. The color limit differs from figure 1; compare coefficients using the bars, not darkness across figures. Zero after the inherited gate does not establish absence of a response.
3. **Leave one strain out.** Left: each held-out strain's centered 13-neuron vector compared with its genus center formed from the remaining strains. Right: that remaining-strain center compared with the full genus center. The 13-genus reference is recomputed on every holdout. One dot per strain; jitter is deterministic solely for visibility. Positive cosine means alignment in this reference-centered coordinate space; it is not a classification score or a test of genus specificity. Offset lengths, within-genus spread, n, and undefined/small-vector flags are in the tables. A large cosine alone does not demonstrate a strong separation, particularly near zero offset.
4. **Same-template pre-gate sensitivity.** Both panels use the same symmetric color scale and primary neural-only order. Each input is centered using its own equal-genus reference. This checks sensitivity to the inherited SNR gate, not template-fitting, date, culture, or sampling uncertainty.
''')
    return {'output':str(out),'genus_summary':summary,'cell_summary':cell_summary,'parameters':params}


def redraw_saved_figures(result_dir, out):
    """Read saved neural tables only, and draw into a fresh output directory.

    Does not recompute scientific summaries, re-read inputs, or alter result_dir.
    Figure files are written under out/figures; an existing nonempty figure
    directory is rejected.
    """
    source = Path(result_dir).resolve()
    destination = Path(out).resolve()
    if destination == source:
        raise FileExistsError('Choose a separate figure output directory; saved results are read-only.')
    figure_dir = destination/'figures'
    if figure_dir.is_dir() and any(figure_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing figures: {figure_dir}')
    read = lambda filename, **kwargs: pd.read_csv(source/'tables'/filename, **kwargs)
    params = json.loads((source/'parameters.json').read_text())
    x = read('strain_unit_profiles.csv', index_col='strain')
    genus = read('main_membership.csv', index_col='strain')['genus'].loc[x.index]
    means = read('genus_mean_unit_profiles.csv', index_col='genus')
    ref = read('equal_genus_reference.csv', index_col='neuron')['reference']
    centered = read('genus_centered_profiles.csv', index_col='genus')
    holdouts = read('leave_one_strain_out.csv')
    pre_centered = read('pre_gate_genus_centered_profiles.csv', index_col='genus')
    figure_dir.mkdir(parents=True, exist_ok=True)
    return plot_results(destination, x, genus, means, ref, centered,
                        params['genus_order'], holdouts, pre_centered)


if __name__=='__main__':
    result=run_analysis(Path(__file__).resolve().parents[4])
    print(result['genus_summary'].to_string())
    print(result['output'])

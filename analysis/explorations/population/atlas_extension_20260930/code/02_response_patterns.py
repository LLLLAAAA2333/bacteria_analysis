"""Exploratory atlas roles and a low-amplitude ASI/ASJ population contrast.

All 13 classes and 78 pairs are screened; ASI/ASJ and post [10,30) s were
selected after that screen. Leave-animal checks do not undo this discovery
selection and are not independent validation. Units: stored calcium dF/F0.
"""
from pathlib import Path
from itertools import combinations
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT / 'reports/exploration_20260929/tables'
T = OUT / 'tables'
NAMES = ['ASK', 'ADL', 'ASI', 'AWA', 'AWB', 'ASG', 'ADF', 'ASH', 'ASJ', 'ASEL', 'ASER', 'AWCON', 'AWCOFF']
KEY = ['sample_id', 'date', 'worm_key', 'neuron_class']
IDX = ['date', 'worm_key', 'sample_id']
PHASES = {'stim': range(0, 10), 'post': range(10, 30), 'full': range(0, 40)}
SEED = 2026093002


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() >= 4 and np.std(x[ok]) and np.std(y[ok]) else np.nan


def build_animals():
    trial = pd.read_parquet(SOURCE / 'trial_curves.parquet').reset_index().sort_values(KEY + ['segment_index'])
    trial['date'] = trial.date.astype(str)
    for phase, time in PHASES.items():
        trial[phase] = trial[[str(t) for t in time]].mean(axis=1)
    trial['baseline'] = trial[[str(t) for t in range(-5, 0)]].mean(axis=1)
    trial['first'] = trial.groupby(KEY).cumcount().eq(0)
    results = []
    for variant, data, agg in [('mean', trial, 'mean'), ('median', trial, 'median'),
                               ('first', trial[trial['first']], 'mean'), ('later', trial[~trial['first']], 'mean')]:
        a = data.groupby(KEY)[[*PHASES, 'baseline', 'segment_index']].agg(agg).reset_index()
        a['variant'] = variant
        results.append(a)
    mean = results[0]
    corrected = mean.copy()
    corrected[list(PHASES)] = corrected[list(PHASES)].sub(corrected.baseline, axis=0)
    corrected['variant'] = 'baseline_subtracted'
    results.append(corrected)
    order = mean.copy()
    # Sensitivity only: detrend each animal's summaries by mean retained segment.
    # Uses all that animal's stimuli, not a deployable new-stimulus predictor.
    for _, ids in order.groupby(['date', 'worm_key', 'neuron_class']).groups.items():
        x = order.loc[ids, 'segment_index'].to_numpy()
        x -= x.mean()
        if x @ x:
            y = order.loc[ids, list(PHASES)].to_numpy()
            order.loc[ids, list(PHASES)] = y - x[:, None] * (x @ y / (x @ x))[None, :]
    order['variant'] = 'order_detrended'
    results.append(order)
    return pd.concat(results, ignore_index=True)


def pair_screen(a, taxonomy):
    records = []
    for phase in PHASES:
        wide = a.pivot(index=IDX, columns='neuron_class', values=phase)
        for left, right in combinations(NAMES, 2):
            q = wide[[left, right]].dropna()
            center = q - q.groupby(['date', 'worm_key']).transform('mean')
            bystrain = center.groupby(['sample_id', 'date']).mean().groupby('sample_id').mean()
            within_genus = bystrain.join(taxonomy.genus_clean)
            within_genus = within_genus[within_genus.groupby('genus_clean').genus_clean.transform('size').ge(3)]
            within_genus[[left, right]] -= within_genus.groupby('genus_clean')[[left, right]].transform('mean')
            per = np.array([rho(g[left], g[right]) for _, g in q.groupby(['date', 'worm_key'])])
            records.append(dict(phase=phase, neuron_a=left, neuron_b=right, n_animals=len(per), n_strains=len(bystrain),
                                n_animal_strain=len(q), centered_strain_rho=rho(bystrain[left], bystrain[right]),
                                n_within_genus_strains=len(within_genus),
                                within_genus_rho=rho(within_genus[left], within_genus[right]),
                                animal_rho_median=np.nanmedian(per), n_positive_animal_rho=int(np.sum(per > 0))))
    return pd.DataFrame(records)


def general_roles(a):
    records = []
    for (variant, neuron), q in a.groupby(['variant', 'neuron_class']):
        s = q.groupby(['sample_id', 'date'])[['stim', 'post', 'full']].mean().groupby('sample_id').mean()
        for threshold in [0, .02, .05]:
            records.append(dict(variant=variant, neuron_class=neuron, deadband=threshold,
                                n_strains=len(s), n_animals=q[['date', 'worm_key']].drop_duplicates().shape[0], n_animal_strain=len(q),
                                median_stim=s.stim.median(), median_post=s.post.median(),
                                n_strain_stim_positive=int((s.stim > threshold).sum()),
                                n_strain_stim_negative=int((s.stim < -threshold).sum()),
                                n_strain_negative_to_positive=int(((s.stim < -threshold) & (s.post > threshold)).sum()),
                                n_strain_persist_negative=int(((s.stim < -threshold) & (s.post < -threshold)).sum()),
                                fraction_animal_strain_negative_to_positive=float(((q.stim < -threshold) & (q.post > threshold)).mean()),
                                fraction_animal_strain_stim_negative=float((q.stim < -threshold).mean())))
    return pd.DataFrame(records)


def coherence_checks(a, taxonomy):
    records, animal_records = [], []
    for variant, data in a.groupby('variant'):
        for phase in PHASES:
            q = data.pivot(index=IDX, columns='neuron_class', values=phase)[['ASI', 'ASJ']].dropna()
            for exclusion in ['none', 'A247', 'top3_abs_score']:
                center = q - q.groupby(['date', 'worm_key']).transform('mean')
                s = center.groupby(['sample_id', 'date']).mean().groupby('sample_id').mean()
                excluded = [] if exclusion == 'none' else ['A247'] if exclusion == 'A247' else s.mean(axis=1).abs().nlargest(3).index.tolist()
                keep = ~q.index.get_level_values('sample_id').isin(excluded)
                z = q[keep]
                center = z - z.groupby(['date', 'worm_key']).transform('mean')
                s = center.groupby(['sample_id', 'date']).mean().groupby('sample_id').mean()
                g = s.join(taxonomy.genus_clean)
                g = g[g.groupby('genus_clean').genus_clean.transform('size').ge(3)]
                g[['ASI', 'ASJ']] -= g.groupby('genus_clean')[['ASI', 'ASJ']].transform('mean')
                per = []
                for (date, worm), obs in z.groupby(['date', 'worm_key']):
                    r = rho(obs.ASI, obs.ASJ)
                    per.append(r)
                    animal_records.append(dict(variant=variant, phase=phase, exclusion=exclusion,
                                               date=date, worm_key=worm, n_strains=len(obs), rho=r))
                records.append(dict(variant=variant, phase=phase, exclusion=exclusion,
                                    excluded_strains=';'.join(excluded), n_animals=len(per), n_strains=len(s),
                                    n_within_genus_strains=len(g),
                                    centered_strain_rho=rho(s.ASI, s.ASJ), within_genus_rho=rho(g.ASI, g.ASJ),
                                    animal_rho_median=np.nanmedian(per), n_positive_animal_rho=int(np.sum(np.array(per) > 0))))
    return pd.DataFrame(records), pd.DataFrame(animal_records)


def held_animal_contrasts(a, curves):
    """Training animals choose top/bottom three stimuli; test animal supplies y.

    Scores are raw-unit mean of the two post responses. No test y is used in
    selection. Both cells and all six selected strains required in each held
    animal. At least two other jointly observed animals per selected strain.
    """
    mean = a[a.variant.eq('mean')].pivot(index=IDX, columns='neuron_class', values='post')
    mean = mean[['ASI', 'ASJ']].dropna()
    variants = {v: g.pivot(index=IDX, columns='neuron_class', values='post') for v, g in a.groupby('variant')}
    records, assignments, raw_curves, crosscell = [], [], [], []
    for (date, worm), test0 in mean.groupby(['date', 'worm_key']):
        test0 = test0.droplevel(['date', 'worm_key'])
        train0 = mean[(mean.index.get_level_values('date') == date) & (mean.index.get_level_values('worm_key') != worm)]
        train = train0.groupby('sample_id').mean()
        count = train0.groupby('sample_id').size()
        common = test0.index.intersection(train.index[count.ge(2)])
        if len(common) < 6:
            continue
        train, test = train.loc[common], test0.loc[common]
        # Cross-neuron transfer tests whether coherence exists across animals,
        # rather than only simultaneous fluctuation in a single animal.
        for selector, target in [('ASI', 'ASJ'), ('ASJ', 'ASI')]:
            crosscell.append(dict(date=date, worm_key=worm, selector=selector, target=target,
                                  n_strains=len(common), rho=rho(train[selector], test[target])))
        score = train.mean(axis=1)
        low, high = score.nsmallest(3).index.tolist(), score.nlargest(3).index.tolist()
        for group, strains in [('higher', high), ('lower', low)]:
            for sid in strains:
                assignments.append(dict(date=date, worm_key=worm, group=group, sample_id=sid,
                                        train_score=score.loc[sid], n_train_animals=count.loc[sid]))
        for variant, wide in variants.items():
            try:
                obs = wide.loc[(date, worm)].reindex(high + low)
            except KeyError:
                continue
            if obs[['ASI', 'ASJ']].isna().any().any():
                continue
            for n in NAMES:
                if n not in obs or obs[n].isna().any():
                    continue
                records.append(dict(date=date, worm_key=worm, variant=variant, neuron_class=n,
                                    high_mean=obs.loc[high, n].mean(), low_mean=obs.loc[low, n].mean(),
                                    difference=obs.loc[high, n].mean() - obs.loc[low, n].mean()))
        for group, strains in [('higher', high), ('lower', low)]:
            for n in ['ASI', 'ASJ', 'ASK', 'ADF', 'ASH']:
                values = curves[(curves.date == date) & (curves.worm_key == worm) &
                                curves.sample_id.isin(strains) & curves.neuron_class.eq(n)]
                if values.sample_id.nunique() != 3:
                    continue
                record = dict(date=date, worm_key=worm, group=group, neuron_class=n,
                              selected_strains=';'.join(strains))
                record.update(values[[str(t) for t in range(-5, 40)]].mean().to_dict())
                raw_curves.append(record)
    return pd.DataFrame(records), pd.DataFrame(assignments), pd.DataFrame(raw_curves), pd.DataFrame(crosscell)


def residual_checks(a):
    rows = []
    for phase in PHASES:
        wide = a[a.variant.eq('mean')].pivot(index=IDX, columns='neuron_class', values=phase)
        center = wide - wide.groupby(['date', 'worm_key']).transform('mean')
        other = center.drop(columns=['ASI', 'ASJ'])
        center['other_raw_mean'] = other.mean(axis=1)
        center['other_scaled_median'] = (other / other.std()).median(axis=1)
        for controls in [['ADF', 'ASH'], ['ADF', 'ASH', 'ASK', 'AWCON'],
                         ['other_raw_mean'], ['other_scaled_median']]:
            z = center[['ASI', 'ASJ'] + controls].dropna()
            # Descriptive competitor adjustment, not causal conditioning or a
            # fitted mechanism. Animal centering precedes control selection.
            x = np.c_[np.ones(len(z)), z[controls].to_numpy()]
            residual = z[['ASI', 'ASJ']].to_numpy() - x @ np.linalg.lstsq(x, z[['ASI', 'ASJ']].to_numpy(), rcond=None)[0]
            r = pd.DataFrame(residual, index=z.index, columns=['ASI', 'ASJ'])
            s = r.groupby(['sample_id', 'date']).mean().groupby('sample_id').mean()
            rows.append(dict(phase=phase, controls=';'.join(controls), n_animals=z.reset_index()[['date', 'worm_key']].drop_duplicates().shape[0],
                             n_animal_strain=len(z), n_strains=len(s),
                             original_rho=rho(z.ASI, z.ASJ), residual_rho=rho(r.ASI, r.ASJ),
                             residual_strain_rho=rho(s.ASI, s.ASJ)))
    return pd.DataFrame(rows)


def within_genus_check(a, taxonomy):
    """Restrict ranking to >=3 strains of the same genus and test high-low.

    Not the same contrast as the catalogue top/bottom-three. Included to ask
    whether the selected five-cell program must hold within a genus.
    """
    wide = a[a.variant.eq('mean')].pivot(index=IDX, columns='neuron_class', values='post')
    selected = wide[['ASI', 'ASJ']].dropna()
    records = []
    for (date, worm), test in selected.groupby(['date', 'worm_key']):
        train = selected[(selected.index.get_level_values('date') == date) & (selected.index.get_level_values('worm_key') != worm)]
        count = train.groupby('sample_id').size()
        train = train.groupby('sample_id').mean()
        common = test.index.get_level_values('sample_id').intersection(train.index[count.ge(2)])
        train = train.loc[common].join(taxonomy.genus_clean)
        observed = wide.loc[(date, worm)].loc[common]
        for genus, subset in train.groupby('genus_clean'):
            if len(subset) < 3:
                continue
            score = subset[['ASI', 'ASJ']].mean(axis=1)
            high, low = score.idxmax(), score.idxmin()
            for n in ['ASI', 'ASJ', 'ASK', 'ADF', 'ASH']:
                if pd.notna(observed.loc[high, n]) and pd.notna(observed.loc[low, n]):
                    records.append(dict(date=date, worm_key=worm, genus_clean=genus, neuron_class=n,
                                        high_strain=high, low_strain=low, n_genus_candidates=len(subset),
                                        difference=observed.loc[high, n] - observed.loc[low, n]))
    return pd.DataFrame(records)


def gain_checks(contrasts):
    records = []
    names = ['ASI', 'ASJ', 'ASK', 'ADF', 'ASH']
    for variant, q in contrasts.groupby('variant'):
        high = q.pivot(index=['date', 'worm_key'], columns='neuron_class', values='high_mean')[names]
        low = q.pivot(index=['date', 'worm_key'], columns='neuron_class', values='low_mean')[names]
        valid = high.notna().all(axis=1) & low.notna().all(axis=1)
        high, low = high[valid], low[valid]
        for (date, worm), h in high.iterrows():
            l = low.loc[(date, worm)].to_numpy()
            h = h.to_numpy()
            hnorm, lnorm = np.linalg.norm(h), np.linalg.norm(l)
            if min(hnorm, lnorm) <= 0:
                continue
            beta = max(0., float(h @ l / (l @ l)))
            record = dict(date=date, worm_key=worm, variant=variant, high_norm=hnorm, low_norm=lnorm,
                          positive_gain=beta, relative_residual=np.linalg.norm(h - beta * l) / hnorm)
            record.update({n + '_normalized_difference': h[i] / hnorm - l[i] / lnorm for i, n in enumerate(names)})
            records.append(record)
    return pd.DataFrame(records)


def plot(curves, contrasts):
    time = np.arange(-5, 40)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=True, layout='constrained')
    colors = {'higher': '#c27623', 'lower': '#2d709a'}
    for ax, neuron in zip(axes, ['ASI', 'ASJ', 'ASK', 'ADF', 'ASH']):
        q = curves[curves.neuron_class.eq(neuron)]
        n = q[['date', 'worm_key']].drop_duplicates().shape[0]
        for group in ['higher', 'lower']:
            y = q[q.group.eq(group)][[str(t) for t in time]].to_numpy()
            for line in y:
                ax.plot(time, line, color=colors[group], lw=.6, alpha=.16)
            ax.plot(time, y.mean(axis=0), color=colors[group], lw=2.4,
                    label=f'{group.capitalize()} training score')
        ax.axvspan(0, 10, color='.85', alpha=.6)
        ax.axhline(0, color='.5', lw=.6)
        check = contrasts[(contrasts.variant == 'mean') & (contrasts.neuron_class == neuron)]
        direction = 'higher' if neuron in ['ASI', 'ASJ', 'ASK'] else 'lower'
        count = int((check.difference > 0).sum()) if direction == 'higher' else int((check.difference < 0).sum())
        ax.set(title=f'{neuron}\nPost response {direction}: {count}/{len(check)} animals',
               xlabel='Seconds after stimulus onset', xlim=(-5, 39))
    axes[0].set_ylabel(r'Measured calcium $\Delta F/F_0$')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle(f'Stimuli ranked using ASI/ASJ in other animals also separate ASK, ADF and ASH\nSame {n} held animals; thin curves: animals; thick curves: means', fontsize=12)
    fig.savefig(OUT / 'figures/response_patterns_asi_asj.png', dpi=190)
    fig.savefig(OUT / 'figures/response_patterns_asi_asj.pdf')
    plt.close(fig)


def main():
    for folder in ['tables', 'logs', 'figures']:
        (OUT / folder).mkdir(exist_ok=True)
    a = build_animals()
    a.to_csv(T / 'response_patterns_animal_metrics.csv', index=False)
    reference = pd.read_csv(SOURCE / 'animal_metrics.csv', dtype={'date': str})
    merged = a[a.variant.eq('mean')].merge(reference, on=KEY, suffixes=('_new', '_old'), validate='one_to_one')
    error = max(abs(merged[f'{w}_new'] - merged[f'{w}_old']).max() for w in PHASES)
    assert error < 1e-12
    taxonomy = pd.read_csv(SOURCE / 'taxonomy.csv').set_index('sample_id')
    screen = pair_screen(a[a.variant.eq('mean')], taxonomy)
    screen.to_csv(T / 'response_patterns_all_pair_screen.csv', index=False)
    general_roles(a).to_csv(T / 'response_patterns_general_roles.csv', index=False)
    checks, animal = coherence_checks(a, taxonomy)
    checks.to_csv(T / 'response_patterns_coherence_checks.csv', index=False)
    animal.to_csv(T / 'response_patterns_coherence_animals.csv', index=False)
    residual_checks(a).to_csv(T / 'response_patterns_adf_ash_adjustment.csv', index=False)
    curves = pd.read_parquet(SOURCE / 'animal_curves.parquet').reset_index()
    curves['date'] = curves.date.astype(str)
    contrasts, assignments, selected_curves, crosscell = held_animal_contrasts(a, curves)
    contrasts.to_csv(T / 'response_patterns_held_contrasts.csv', index=False)
    assignments.to_csv(T / 'response_patterns_held_assignments.csv', index=False)
    selected_curves.to_csv(T / 'response_patterns_held_curves.csv', index=False)
    crosscell.to_csv(T / 'response_patterns_crosscell_transfer.csv', index=False)
    assignments.join(taxonomy[['genus_clean', 'species_clean']], on='sample_id').to_csv(T / 'response_patterns_held_taxonomy.csv', index=False)
    assignments.join(taxonomy[['genus_clean']], on='sample_id').groupby(['group', 'genus_clean']).agg(
        selected_appearances=('sample_id', 'size'), n_strains=('sample_id', 'nunique'),
        n_blocks=('date', 'nunique')).reset_index().to_csv(T / 'response_patterns_selected_genera.csv', index=False)
    assignments.groupby(['group', 'sample_id']).agg(selected_appearances=('worm_key', 'size'), n_blocks=('date', 'nunique')).reset_index().join(
        taxonomy[['genus_clean', 'species_clean']], on='sample_id').to_csv(T / 'response_patterns_selected_strains.csv', index=False)
    summary = contrasts.groupby(['variant', 'neuron_class']).difference.agg(
        mean='mean', median='median', minimum='min', maximum='max', n_animals='size',
        n_positive=lambda x: int((x > 0).sum())).reset_index()
    summary.to_csv(T / 'response_patterns_held_summary.csv', index=False)
    five = ['ASI', 'ASJ', 'ASK', 'ADF', 'ASH']
    wide = contrasts.pivot(index=['date', 'worm_key', 'variant'], columns='neuron_class', values='difference')[five].dropna()
    wide.to_csv(T / 'response_patterns_matched_five_contrasts.csv')
    wide.groupby('variant').agg(['mean', 'median', 'min', 'max', 'size', lambda x: int((x > 0).sum())]).to_csv(T / 'response_patterns_matched_five_summary.csv')
    matching = wide.xs('mean', level='variant').index
    matched_curves = selected_curves[pd.MultiIndex.from_frame(selected_curves[['date', 'worm_key']]).isin(matching)]
    matched_curves.to_csv(T / 'response_patterns_matched_five_curves.csv', index=False)
    matched_contrasts = contrasts[pd.MultiIndex.from_frame(contrasts[['date', 'worm_key']]).isin(matching)]
    plot(matched_curves, matched_contrasts)
    matching_assignments = assignments[pd.MultiIndex.from_frame(assignments[['date', 'worm_key']]).isin(matching)]
    matching_assignments.join(taxonomy[['genus_clean', 'species_clean']], on='sample_id').to_csv(
        T / 'response_patterns_matched_five_assignments.csv', index=False)
    matching_assignments.join(taxonomy[['genus_clean']], on='sample_id').groupby(['group', 'genus_clean']).agg(
        selected_appearances=('sample_id', 'size'), n_strains=('sample_id', 'nunique'), n_blocks=('date', 'nunique')).reset_index().to_csv(
            T / 'response_patterns_matched_five_genera.csv', index=False)
    gain_checks(contrasts).to_csv(T / 'response_patterns_gain_checks.csv', index=False)
    byblock = wide.xs('mean', level='variant').groupby('date').mean()
    byblock.to_csv(T / 'response_patterns_block_contrasts_internal.csv')
    leaveblock = []
    for block in byblock.index:
        obs = wide.xs('mean', level='variant')
        left = obs[obs.index.get_level_values('date') != block]
        for neuron in five:
            leaveblock.append(dict(excluded_block=block, neuron_class=neuron, n_animals=len(left),
                                   mean_difference=left[neuron].mean(), n_positive=int((left[neuron] > 0).sum())))
    pd.DataFrame(leaveblock).to_csv(T / 'response_patterns_leave_block_out.csv', index=False)
    within = within_genus_check(a, taxonomy)
    within.to_csv(T / 'response_patterns_within_genus_contrasts.csv', index=False)
    within.groupby(['genus_clean', 'neuron_class']).difference.agg(['mean', 'median', 'size', lambda x: int((x > 0).sum())]).to_csv(T / 'response_patterns_within_genus_summary.csv')
    methods = dict(seed=SEED, run_status='success', input_reconstruction_max_error=error,
                   discovery='All 13 classes, 78 pairs, stim/post/full screened; ASI/ASJ post chosen from screen. No confirmatory p values.',
                   observation='One animal-strain-cell after trial averaging. Whole (date,worm_key) is repeat unit.',
                   held_animal='Other jointly observed animals in same block rank ASI/ASJ raw-mean post response; fixed three highest and three lowest, then evaluate held animal. At least two training animals per strain.',
                   scope='Conditional on measured stimulus catalogues and shared sequences, no new-strain or cross-date validation.',
                   deadband='0, 0.02, 0.05 dF/F0 are descriptive sensitivity cutoffs, not calibrated event thresholds.',
                   limitations='Different higher/lower strain sets across blocks; common linear drift checked but nonlinear order/carryover and calcium recording artifacts remain. No inference of anatomical connectivity, firing or behavior.',
                   baseline_caveat='Baseline subtraction is algebraic only: stored pre-stimulus means already zero from upstream processing, and this does not constitute another successful falsification.',
                   input_sha256={str(SOURCE / n): hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() for n in ['trial_curves.parquet', 'animal_curves.parquet', 'animal_metrics.csv', 'taxonomy.csv']})
    (OUT / 'logs/response_patterns_methods.json').write_text(json.dumps(methods, indent=2))
    print(checks[(checks.phase == 'post') & (checks.exclusion != 'top3_abs_score')].round(4).to_string(index=False))
    print(summary[summary.neuron_class.isin(['ASI', 'ASJ'])].round(4).to_string(index=False))
    print(crosscell.groupby(['selector', 'target']).rho.agg(['mean', 'median', 'size', lambda x: int((x > 0).sum())]).to_string())


if __name__ == '__main__':
    main()

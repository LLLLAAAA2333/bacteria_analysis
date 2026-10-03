"""Cross-animal paired response differences in a fixed 13-neuron coordinate system.

Observations: trial-averaged animal x strain x cell x 5 calcium bins (0--25 s).
Unit of replication: distinct animals within the recorded acquisition block.
The statistic excludes every animal's self product, keeps negative estimates,
and is not cross-date validation, crossnobis, or a causal strain effect.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SRC = ROOT / 'reports/population_first_20260930/tables'
T, L = OUT / 'tables', OUT / 'logs'
NEURONS = ['ASK', 'ADL', 'ASI', 'AWA', 'AWB', 'ASG', 'ADF', 'ASH',
           'ASJ', 'ASEL', 'ASER', 'AWCON', 'AWCOFF']
MIN_ANIMALS = 3


def cross_animal(d):
    """Average inner product of DISTINCT animals, first averaged over 5 bins."""
    n = len(d)
    if n < 2:
        return np.nan
    return float(np.mean((d.sum(axis=0)**2 - (d*d).sum(axis=0)) / (n*(n-1))))


def cell_scales(wide, columns):
    """Pooled-bin SD of animal-centered strain/block means; no missing fill."""
    centered = wide[columns] - wide.groupby(['date', 'worm_key'])[columns].transform('mean')
    centered = pd.concat([wide[['sample_id', 'date']], centered], axis=1)
    means = centered.groupby(['sample_id', 'date'])[columns].mean()
    rows = []
    for cell in NEURONS:
        cc = [c for c in columns if c.startswith(cell + '__')]
        q = means[cc].dropna()
        observations = wide[cc].notna().all(axis=1)
        scale = np.sqrt(q.var(ddof=1).mean()) if len(q) >= 3 else np.nan
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nan
        rows.append(dict(neuron=cell, scale=scale, n_strain_blocks=len(q),
                         n_strains=wide.loc[observations, 'sample_id'].nunique(),
                         n_animals=wide.loc[observations, ['date', 'worm_key']].drop_duplicates().shape[0],
                         n_animal_strain_rows=int(observations.sum())))
    return pd.DataFrame(rows).set_index('neuron')


def main():
    T.mkdir(parents=True, exist_ok=True)
    L.mkdir(parents=True, exist_ok=True)
    inputs = [SRC/'aligned_neural_animal_5bins.parquet', SRC/'neural_value_pairs.csv',
              SRC/'population_structure_loadings.csv']
    x = pd.read_parquet(inputs[0]).reset_index()
    x['date'] = x.date.astype(str)
    pairs = pd.read_csv(inputs[1], dtype={'date': str})
    columns = [c for c in x if '__' in c]
    assert len(columns) == 65 and x.duplicated(['sample_id', 'date', 'worm_key']).sum() == 0
    fixed = cell_scales(x, columns)
    previous = pd.read_csv(inputs[2]).query('n_bins == 5 and variant == "cell_scaled" and axis == 1')
    previous = previous.groupby('neuron_class').cell_scale.first().reindex(NEURONS)
    fixed_delta = float(np.max(np.abs(fixed.scale.to_numpy()-previous.to_numpy())))
    assert fixed_delta < 1e-12, fixed_delta
    fixed.reset_index().to_csv(T/'pair_signal_fixed_scales.csv', index=False)
    percell, animal_rows, exclusion_rows = [], [], []
    pair_differences = {}
    max_formula_error = 0.
    max_identity_error = 0.
    for p in pairs.itertuples(index=False):
        block = x[x.date.eq(p.date)]
        a = block[block.sample_id.eq(p.strain_a)].set_index('worm_key')[columns]
        b = block[block.sample_id.eq(p.strain_b)].set_index('worm_key')[columns]
        common = a.index.intersection(b.index).sort_values()
        differences = a.loc[common]-b.loc[common]
        pair_differences[p.pair_id] = differences
        other = block[~block.sample_id.isin([p.strain_a, p.strain_b])]
        excluded = cell_scales(other, columns)
        excluded['pair_id'] = p.pair_id
        exclusion_rows.append(excluded.reset_index())
        for cell in NEURONS:
            cc = [c for c in columns if c.startswith(cell+'__')]
            q = differences[cc].dropna()
            d = q.to_numpy()
            n = len(d)
            eligible = n >= MIN_ANIMALS
            u = cross_animal(d) if eligible else np.nan
            mu = d.mean(axis=0) if n else np.full(5, np.nan)
            mean_square = float(np.mean(mu**2)) if n else np.nan
            noise_mean = float(np.mean(d.var(axis=0, ddof=1)/n)) if n >= 2 else np.nan
            if eligible:
                explicit = np.mean([np.mean(d[i]*d[j]) for i in range(n) for j in range(n) if i != j])
                max_formula_error = max(max_formula_error, abs(u-explicit))
                max_identity_error = max(max_identity_error, abs(u-(mean_square-noise_mean)))
            sf, se = fixed.loc[cell, 'scale'], excluded.loc[cell, 'scale']
            percell.append(dict(pair_id=p.pair_id, date=p.date, strain_a=p.strain_a,
                strain_b=p.strain_b, neuron=cell, n_animals=n, eligible=eligible,
                energy_raw=u, energy_fixed=u/sf**2, energy_excluded=u/se**2,
                fixed_scale=sf, excluded_scale=se, signed_mean_raw=float(mu.mean()),
                mean_square_raw=mean_square, noise_of_mean_raw=noise_mean,
                positive_mean_bins=int((mu>0).sum()),
                excluded_n_strains=int(excluded.loc[cell, 'n_strains']),
                excluded_n_animals=int(excluded.loc[cell, 'n_animals'])))
            for worm, values in zip(q.index, d):
                for j, value in enumerate(values):
                    animal_rows.append(dict(pair_id=p.pair_id, date=p.date, worm_key=worm,
                        strain_a=p.strain_a, strain_b=p.strain_b, neuron=cell,
                        bin=j, start_s=j*5, stop_s_exclusive=j*5+5,
                        difference_raw=float(value), difference_fixed=float(value/sf)))
    pc = pd.DataFrame(percell)
    counts = pc.pivot(index='pair_id', columns='neuron', values='n_animals')
    common_cells = [n for n in NEURONS if counts[n].min() >= MIN_ANIMALS]
    assert len(common_cells) == 10, 'Update sensitivity field names if the common panel changes.'
    rows, deletion = [], []
    for p in pairs.itertuples(index=False):
        q = pc[pc.pair_id.eq(p.pair_id)].set_index('neuron')
        used = q[q.eligible].index.tolist()
        d = pair_differences[p.pair_id]
        r = dict(p._asdict())
        r.update(n_cells=len(used), all_13_cells=len(used)==13,
                 n_paired_animals_min=int(q.loc[used, 'n_animals'].min()),
                 n_paired_animals_max=int(q.loc[used, 'n_animals'].max()),
                 n_animals_any=int(d.notna().any(axis=1).sum()))
        for variant in ['raw', 'fixed', 'excluded']:
            vals = q.loc[used, 'energy_'+variant]
            r['energy_'+variant] = float(vals.mean()) if vals.notna().all() else np.nan
            cvals = q.loc[common_cells, 'energy_'+variant]
            r['energy_common10_'+variant] = float(cvals.mean()) if cvals.notna().all() else np.nan
        r['mean_square_fixed'] = float(np.mean(q.loc[used, 'mean_square_raw']/q.loc[used, 'fixed_scale']**2))
        r['noise_of_mean_fixed'] = float(np.mean(q.loc[used, 'noise_of_mean_raw']/q.loc[used, 'fixed_scale']**2))
        # A second missingness sensitivity uses only animals complete for ALL 65 entries.
        complete = d.dropna()
        r['n_complete13_animals'] = len(complete)
        r['energy_complete13_animals_fixed'] = (cross_animal(complete.to_numpy()/np.repeat(fixed.scale.to_numpy(), 5))
            if len(complete) >= MIN_ANIMALS else np.nan)
        for animal in d.index:
            values = {'raw': [], 'fixed': [], 'excluded': []}
            for cell in used:
                cc = [c for c in columns if c.startswith(cell+'__')]
                z = d.drop(index=animal)[cc].dropna().to_numpy()
                # Keep the original cell panel: after deletion n may be 2, still a distinct-animal statistic.
                assert len(z) >= 2
                u = cross_animal(z)
                values['raw'].append(u)
                values['fixed'].append(u/q.loc[cell, 'fixed_scale']**2)
                values['excluded'].append(u/q.loc[cell, 'excluded_scale']**2)
            deletion.append(dict(pair_id=p.pair_id, date=p.date, deleted_worm=animal,
                n_cells=len(used), **{'energy_'+v: float(np.mean(z)) for v,z in values.items()}))
        dz = pd.DataFrame(deletion).query('pair_id == @p.pair_id')
        for variant in ['raw', 'fixed', 'excluded']:
            r['loo_'+variant+'_min'] = float(dz['energy_'+variant].min())
            r['loo_'+variant+'_max'] = float(dz['energy_'+variant].max())
        rows.append(r)
    summary = pd.DataFrame(rows)
    pc = pc.merge(summary[['pair_id', 'n_cells']], on='pair_id', validate='many_to_one')
    pc['contribution_fixed'] = pc.energy_fixed/pc.n_cells
    pc['contribution_raw'] = pc.energy_raw/pc.n_cells
    pc.to_csv(T/'pair_signal_percell.csv', index=False)
    pd.DataFrame(animal_rows).to_csv(T/'pair_signal_animal_differences.csv', index=False)
    pd.concat(exclusion_rows, ignore_index=True).to_csv(T/'pair_signal_excluded_scales.csv', index=False)
    summary.to_csv(T/'pair_signal_summary.csv', index=False)
    pd.DataFrame(deletion).to_csv(T/'pair_signal_animal_deletion.csv', index=False)
    assert len(summary) == 147 and len(pc) == 147*13
    assert max_formula_error < 1e-12 and max_identity_error < 1e-12
    assert np.allclose(summary.energy_fixed, summary.mean_square_fixed-summary.noise_of_mean_fixed)
    assert not pc.loc[~pc.eligible, 'energy_fixed'].notna().any()
    facts = dict(status='success', n_pairs=len(summary), n_full13_pairs=int(summary.all_13_cells.sum()),
        n_strains=len(set(summary.strain_a)|set(summary.strain_b)),
        n_cells_histogram=summary.n_cells.value_counts().sort_index().to_dict(),
        n_blocks=int(summary.date.nunique()), common_cells=common_cells,
        fixed_scale_vs_previous_max_abs_error=fixed_delta,
        cross_product_explicit_vs_formula_max_abs_error=max_formula_error,
        mean_square_minus_noise_vs_formula_max_abs_error=max_identity_error,
        fixed_negative_count=int((summary.energy_fixed<0).sum()),
        full13_fixed_negative_count=int((summary.loc[summary.all_13_cells, 'energy_fixed']<0).sum()),
        full13_fixed_quantiles=summary.loc[summary.all_13_cells, 'energy_fixed'].quantile([0,.25,.5,.75,1]).to_dict(),
        n_pairs_complete13_animal_sensitivity=int(summary.energy_complete13_animals_fixed.notna().sum()),
        n_pairs_missing_excluded_scale=int(summary.energy_excluded.isna().sum()),
        inputs=[dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in inputs])
    (L/'pair_signal_verification.json').write_text(json.dumps(facts, indent=2)+'\n')
    methods = '''# Pair response statistic\n\nFor each pre-existing within-block, same-genus, same-chemical-reference strain pair, define d[a,c,b] as strain A minus B calcium response in animal a, neuron c and 5-second bin b. Trial means and bilateral cell merging are inherited from the verified aligned table. Five bins cover 0--25 s after stimulus onset; stimulus delivery was 0--10 s.\n\nFor a neuron measured for both strains in at least three animals, U[c] = mean over the five bins of {sum(a != a') d[a,c,b] d[a',c,b] / [n(c)(n(c)-1)]}. Every self product is excluded. This is exactly mean_b(mean_a d squared) minus mean_b(sample variance across animals / n). The latter is an algebraic identity, not an assertion of noise-unbiasedness: other-stimulus scaling, fixed order, shared blocks and animal dependence remain possible. Negative estimates are retained and indicate insufficient reproducible directional difference, not proof of equal responses.\n\nMain fixed scaling is one scalar per neuron, reused from population-first analysis and numerically reconstructed: subtract each animal's mean over its presented strains separately for each neuron/bin; average animals within each strain/block; take the sample SD across the 112 strain/block means separately per bin; use the square root of the mean of those five variances. Pair scores are mean of U[c]/scale[c]^2 across eligible cells, so bins then neurons receive equal weight. This common coordinate system is descriptive, estimated once from this catalogue, not held-out calibration. The complete 13-cell subset contains 137 pairs. Ten other pairs have only 10 eligible cells. All 147 are retained in the main table, with explicit coverage markers and missing cells shown as missing. The 147-pair fixed common 10-cell sensitivity checks unequal coverage; no pair is discarded based on the strength of its response.\n\nSensitivity 1: raw dF/F0 squared. Sensitivity 2: recompute the same cell scale within the pair's block after removing BOTH pair strains, including re-centering animals over remaining strains; require three other strain/block means. This removes direct response-dependent scaling by the pair, but differs between pairs and is not an independent/noise-unbiased metric. Support is exported. Sensitivity 3: same 10 cells supported across all 147 pairs. Sensitivity 4: only animals complete in all 13 neurons; require at least 3 such animals.\n\nDelete-one-animal ranges remove an entire animal's paired differences in every neuron/bin together, retaining the original eligible neuron panel and fixed scales. Cells may then have 2 animals, still permitting distinct-animal products. These min/max ranges are sensitivity ranges, not confidence intervals. No trials, bins, cells, overlapping pairs or within-block pairs are counted as independent biological replicates. No p-values, near/far threshold selection, or causal/date-generalization claim is made.\n\nAll 147 pairs were inherited from chemical/genus/block/reference eligibility, without selection using neural outcomes. Their chemical RMS log2FC distance, nearest-neighbor flags and metadata are retained exactly. The 137 complete-panel pair restriction is exclusively a measurement-coverage rule.\n'''
    (L/'pair_signal_methods.md').write_text(methods)
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()

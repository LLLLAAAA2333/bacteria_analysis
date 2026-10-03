"""Which cells express chemistry-neighbor response differences?

All 13 classes enter together. Positive-part contribution normalization is
descriptive and never alters the signed population statistic. Presentation
transfer excludes the entire target animal from selection AND scale fitting.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
T, L = OUT / 'tables', OUT / 'logs'
NEURONS = ['ASK', 'ADL', 'ASI', 'AWA', 'AWB', 'ASG', 'ADF', 'ASH',
           'ASJ', 'ASEL', 'ASER', 'AWCON', 'AWCOFF']
COLS = [f'{n}__{5*j:02d}_{5*j+5:02d}' for n in NEURONS for j in range(5)]
spec = importlib.util.spec_from_file_location('pair_signal', OUT/'code/01_pair_signal.py')
signal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signal)


def presentation_tables():
    path = ROOT/'reports/exploration_20260929/tables/trial_curves.parquet'
    trial = pd.read_parquet(path).reset_index().sort_values('segment_index')
    trial['date'] = trial.date.astype(str)
    keys = ['sample_id', 'date', 'worm_key', 'neuron_class']
    trial['rank'] = trial.groupby(keys).cumcount()
    bins = [f'bin{j}' for j in range(5)]
    for j, name in enumerate(bins):
        trial[name] = trial[[str(k) for k in range(j*5, (j+1)*5)]].mean(axis=1)
    tables = {}
    for mode, subset in [('first', trial[trial['rank'].eq(0)]),
                         ('later', trial[trial['rank'].gt(0)])]:
        long = subset.groupby(keys)[bins].mean()
        wide = long.unstack('neuron_class')
        wide.columns = [f'{n}__{5*int(b[-1]):02d}_{5*int(b[-1])+5:02d}' for b, n in wide.columns]
        tables[mode] = wide.reindex(columns=COLS).reset_index()
    return path, tables


def main():
    pc = pd.read_csv(T/'pair_signal_percell.csv', dtype={'date': str})
    pairs = pd.read_csv(T/'pair_signal_summary.csv', dtype={'date': str})
    assert pc.shape[0] == 147*13 and not pc.duplicated(['pair_id', 'neuron']).any()
    rows, shares = [], []
    for p in pairs.itertuples(index=False):
        q = pc[pc.pair_id.eq(p.pair_id) & pc.eligible].set_index('neuron')
        e = q.energy_fixed
        positive = e.clip(lower=0)
        prop = positive / positive.sum() if positive.sum() > 0 else positive*np.nan
        top = e.idxmax()
        without = e.drop(index=top).mean()
        r = p._asdict()
        r.update(top_cell=top, top_cell_energy=float(e.loc[top]),
                 top_positive_fraction=float(prop.loc[top]),
                 positive_cell_count=int(e.gt(0).sum()),
                 positive_effective_cell_count=float(1/(prop*prop).sum()),
                 positive_energy_sum=float(positive.sum()),
                 negative_energy_sum=float(e.clip(upper=0).sum()),
                 energy_fixed_without_top=float(without),
                 top_raw_cell=q.energy_raw.idxmax())
        rows.append(r)
        for n in q.index:
            shares.append(dict(pair_id=p.pair_id, neuron=n,
                positive_fraction=prop.loc[n], signed_energy=e.loc[n],
                population_contribution=e.loc[n]/len(e), is_top=n==top))
    summary = pd.DataFrame(rows)
    summary.to_csv(T/'neuron_patterns_pairs.csv', index=False)
    pd.DataFrame(shares).to_csv(T/'neuron_patterns_contributions.csv', index=False)

    trial_path, wide = presentation_tables()
    animals = wide['first'][['date', 'worm_key']].drop_duplicates()
    scales, scale_rows = {}, []
    for mode, w in wide.items():
        for a in animals.itertuples(index=False):
            held = w.date.eq(a.date) & w.worm_key.eq(a.worm_key)
            training = w.loc[~held]
            assert not ((training.date.eq(a.date)) & (training.worm_key.eq(a.worm_key))).any()
            s = signal.cell_scales(training, COLS)
            scales[(mode, a.date, a.worm_key)] = s.scale
            for n in NEURONS:
                scale_rows.append(dict(train_presentation=mode, held_date=a.date,
                    held_worm=a.worm_key, neuron=n, scale=float(s.loc[n, 'scale']),
                    n_train_animals=int(s.loc[n, 'n_animals'])))
    pd.DataFrame(scale_rows).to_csv(T/'neuron_patterns_transfer_scales.csv', index=False)

    folds, cells = [], []
    for p in pairs.itertuples(index=False):
        diffs = {}
        for mode, w in wide.items():
            block = w[w.date.eq(p.date)]
            a = block[block.sample_id.eq(p.strain_a)].set_index('worm_key')[COLS]
            b = block[block.sample_id.eq(p.strain_b)].set_index('worm_key')[COLS]
            common = a.index.intersection(b.index)
            diffs[mode] = a.loc[common]-b.loc[common]
        for train_mode, test_mode in [('first', 'later'), ('later', 'first')]:
            source, target = diffs[train_mode], diffs[test_mode]
            animals_here = source.index.intersection(target.index)
            for animal in animals_here:
                sf = scales[(train_mode, p.date, animal)]
                candidates = []
                for n in NEURONS:
                    cc = [c for c in COLS if c.startswith(n+'__')]
                    train = source.drop(index=animal)[cc].dropna()
                    test = target.loc[animal, cc].to_numpy(float)
                    if len(train) < 2 or not np.isfinite(test).all() or not np.isfinite(sf.loc[n]):
                        continue
                    assert animal not in train.index
                    selection = signal.cross_animal(train.to_numpy())/sf.loc[n]**2
                    agreement = np.mean(train.mean().to_numpy()*test)/sf.loc[n]**2
                    candidates.append(dict(pair_id=p.pair_id, date=p.date, held_worm=animal,
                        train_mode=train_mode, test_mode=test_mode, neuron=n,
                        n_train_animals=len(train), train_animals='|'.join(map(str, train.index)),
                        train_energy=float(selection), held_alignment=float(agreement),
                        scale=float(sf.loc[n])))
                if len(candidates) < 2:
                    continue
                c = pd.DataFrame(candidates).set_index('neuron')
                top = c.train_energy.idxmax()
                for row in candidates:
                    row['is_selected'] = row['neuron']==top
                    cells.append(row)
                folds.append(dict(pair_id=p.pair_id, date=p.date, held_worm=animal,
                    train_mode=train_mode, test_mode=test_mode, n_cells=len(c),
                    selected_cell=top, selected_train_energy=float(c.loc[top, 'train_energy']),
                    selected_held_alignment=float(c.loc[top, 'held_alignment']),
                    all_cell_alignment=float(c.held_alignment.mean()),
                    without_selected_alignment=float(c.drop(index=top).held_alignment.mean())))
    folds = pd.DataFrame(folds)
    cells = pd.DataFrame(cells)
    folds.to_csv(T/'neuron_patterns_transfer_folds.csv', index=False)
    cells.to_csv(T/'neuron_patterns_transfer_cells.csv', index=False)
    transfer = folds.groupby(['pair_id', 'train_mode', 'test_mode'], as_index=False).agg(
        n_held_animals=('held_worm', 'size'), n_cells_min=('n_cells', 'min'), n_cells_max=('n_cells', 'max'),
        selected_held_alignment=('selected_held_alignment', 'mean'),
        all_cell_alignment=('all_cell_alignment', 'mean'),
        without_selected_alignment=('without_selected_alignment', 'mean'),
        selected_positive_fraction=('selected_held_alignment', lambda z: float((z > 0).mean())),
        n_distinct_selected_cells=('selected_cell', 'nunique'))
    transfer.to_csv(T/'neuron_patterns_transfer_pairs.csv', index=False)

    subset_masks = {'complete13': summary.all_13_cells,
                    'nearest_complete13': summary.all_13_cells & summary.nearest_either.eq(1),
                    'nearest_nonautomatic_complete13': summary.all_13_cells & summary.nearest_either.eq(1) & summary.group_n_strains.ge(3),
                    'chemical_first_quartile_complete13': summary.all_13_cells & summary.closest_quartile.eq(1)}
    description, distribution = [], []
    for label, mask in subset_masks.items():
        s = summary[mask]
        base = dict(subset=label, n_pairs=len(s), n_strains=len(set(s.strain_a)|set(s.strain_b)),
            n_blocks=s.date.nunique(), top_positive_fraction_median=s.top_positive_fraction.median(),
            top_positive_fraction_min=s.top_positive_fraction.min(), top_positive_fraction_max=s.top_positive_fraction.max(),
            positive_effective_cell_count_median=s.positive_effective_cell_count.median(),
            positive_effective_cell_count_min=s.positive_effective_cell_count.min(),
            positive_effective_cell_count_max=s.positive_effective_cell_count.max(),
            top_cell_n_unique=s.top_cell.nunique(), n_positive_population=int(s.energy_fixed.gt(0).sum()),
            n_positive_after_remove_top=int(s.energy_fixed_without_top.gt(0).sum()),
            top_raw_cell_n_unique=s.top_raw_cell.nunique(),
            n_same_top_raw_fixed=int(s.top_cell.eq(s.top_raw_cell).sum()))
        for n in NEURONS:
            distribution.append(dict(subset=label, neuron=n, top_pairs=int(s.top_cell.eq(n).sum()),
                top_raw_pairs=int(s.top_raw_cell.eq(n).sum()), n_pairs=len(s)))
        for mode in ['first', 'later']:
            z = transfer[transfer.pair_id.isin(s.pair_id) & transfer.train_mode.eq(mode)]
            r = base | dict(train_mode=mode, n_transfer_pairs=len(z),
                transfer_selected_alignment_positive_pairs=int(z.selected_held_alignment.gt(0).sum()),
                transfer_population_alignment_positive_pairs=int(z.all_cell_alignment.gt(0).sum()),
                transfer_without_selected_positive_pairs=int(z.without_selected_alignment.gt(0).sum()),
                transfer_selected_alignment_median=z.selected_held_alignment.median(),
                transfer_without_selected_alignment_median=z.without_selected_alignment.median())
            description.append(r)
    desc = pd.DataFrame(description)
    desc.to_csv(T/'neuron_patterns_summary.csv', index=False)
    pd.DataFrame(distribution).to_csv(T/'neuron_patterns_top_cells.csv', index=False)
    assert np.allclose(summary.energy_fixed, summary.set_index('pair_id').index.map(
        pc.groupby('pair_id').energy_fixed.mean()))
    assert all(str(r.held_worm) not in r.train_animals.split('|') for r in cells.itertuples())
    assert np.isfinite(folds[['all_cell_alignment', 'without_selected_alignment', 'selected_held_alignment']]).all().all()
    facts = dict(status='success', n_pairs=len(summary), n_transfer_fold_rows=len(folds),
        n_transfer_cell_rows=len(cells), n_scale_rows=len(scale_rows),
        n_held_animals=len(animals), no_held_animal_in_selection_or_scale=True,
        transfer_panel_cells_min=int(folds.n_cells.min()), transfer_panel_cells_max=int(folds.n_cells.max()),
        inputs=[dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                for f in [T/'pair_signal_percell.csv', T/'pair_signal_summary.csv', trial_path, OUT/'code/01_pair_signal.py']])
    (L/'neuron_patterns.json').write_text(json.dumps(facts, indent=2)+'\n')
    print(desc.to_string(index=False))
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()

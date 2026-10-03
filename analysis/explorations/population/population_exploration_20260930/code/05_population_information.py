"""Does cell identity add information beyond total response magnitude?

Nearest-centroid identification among the 11-13 stimuli encountered in the same
sampling block. One whole animal, all its strains and cells, is held out. This
asks about animal-to-animal stimulus discrimination conditional on the sampled
blocks; it is NOT a new-date or new-strain prediction claim. Trials and cells
are never independent replicates. All training choices use training animals.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
PREV = ROOT / 'reports/exploration_20260929/tables'
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
SEED = 2026093005


def predict(train, test, labels, cols, mode):
    """Arrays animal x strain x feature, test strain x feature."""
    tr = train[:, :, cols].copy()
    te = test[:, cols].copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(tr, axis=(0,1))
        scale = np.nanstd(tr, axis=(0,1))
    scale = np.maximum(scale, 1e-6)
    if mode in ['scaled', 'shape', 'rank1', 'scaled_magnitude']:
        tr = (tr - mean) / scale
        te = (te - mean) / scale
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        centroid = np.nanmean(tr, axis=0)
    if mode in ['magnitude','scaled_magnitude']:
        # Compute the norm of the same mean population template used above.
        # Taking each partly observed training animal's norm first would mix
        # different cell sets into the scalar control.
        centroid = np.sqrt(np.mean(centroid**2, axis=1, keepdims=True))
        te = np.sqrt(np.mean(te**2, axis=1, keepdims=True))
    elif mode in ['shape','raw_shape']:
        # Normalize the same population template as the corresponding raw or
        # scaled model, avoiding norms of partially observed training animals.
        # Only raw_shape is invariant to positive gain about physical zero.
        centroid /= np.maximum(np.sqrt(np.mean(centroid**2,axis=1,keepdims=True)),1e-12)
        te /= np.maximum(np.sqrt(np.mean(te**2,axis=1,keepdims=True)),1e-12)
    elif mode == 'signed_mean':
        centroid = np.mean(centroid, axis=1, keepdims=True)
        te = np.mean(te, axis=1, keepdims=True)
    if mode == 'rank1':
        _, _, vh = np.linalg.svd(centroid - centroid.mean(axis=0), full_matrices=False)
        centroid = centroid @ vh[0, :, None]
        te = te @ vh[0, :, None]
    # Candidate-specific missing coordinates could leak identities. Restrict
    # to coordinates observed in *all* candidate centroids and all test rows.
    valid = np.isfinite(centroid).all(axis=0) & np.isfinite(te).all(axis=0)
    assert valid.any()
    distance = np.mean((te[:, None, valid] - centroid[None, :, valid])**2, axis=2)
    return np.array(labels)[distance.argmin(axis=1)]


def main():
    a = pd.read_csv(PREV / 'animal_metrics.csv')
    a['pooled30']=(10*a.stim+20*a.post)/30
    outputs = []
    choices = []
    for window in ['stim', 'post', 'both', 'full', 'pooled30']:
        windows = ['stim','post'] if window == 'both' else [window]
        features = [(w,n) for w in windows for n in NEURONS]
        for date, d in a.groupby('date'):
            animals = sorted(d.worm_key.unique())
            strains = sorted(d.sample_id.unique())
            values = np.full((len(animals),len(strains),len(features)), np.nan)
            for ai, animal in enumerate(animals):
                sub = d[d.worm_key.eq(animal)].set_index(['sample_id','neuron_class'])
                for fi,(w,n) in enumerate(features):
                    values[ai,:,fi] = sub[w].reindex(pd.MultiIndex.from_product([strains,[n]])).to_numpy()
            for test_i, animal in enumerate(animals):
                tr = np.delete(values, test_i, axis=0)
                te = values[test_i]
                mask = np.isfinite(te).all(axis=0) & (np.isfinite(tr).sum(axis=0)>=2).all(axis=0)
                cols = np.flatnonzero(mask)
                assert len(cols) >= 4, (date, animal, window, len(cols))
                # Select one cell using inner held-out training animals. For
                # both phases, the selected cell has both phase coordinates.
                candidate_scores = {}
                cell_cols = {}
                for n in NEURONS:
                    cc = [c for c in cols if features[c][1] == n]
                    if len(cc) != len(windows):
                        continue
                    inner = []
                    for inner_i in range(len(tr)):
                        it = np.delete(tr, inner_i, axis=0)
                        iv = tr[inner_i]
                        if not np.isfinite(iv[:,cc]).all() or not np.isfinite(it[:,:,cc]).any(axis=0).all():
                            continue
                        pr = predict(it, iv, strains, cc, 'scaled')
                        inner.extend(pr == strains)
                    if inner:
                        candidate_scores[n] = float(np.mean(inner))
                        cell_cols[n] = cc
                assert candidate_scores
                best = max(candidate_scores, key=lambda n: (candidate_scores[n], -NEURONS.index(n)))
                choices.append(dict(window=window,date=date,worm_key=animal, selected_cell=best,
                                    inner_accuracy=candidate_scores[best],n_features=len(cols),
                                    n_train_animals=len(tr),n_stimuli=len(strains)))
                for mode in ['raw','scaled','shape','raw_shape','magnitude','scaled_magnitude','signed_mean','rank1','best_cell','without_best_cell']:
                    if mode == 'best_cell':
                        cc, method = cell_cols[best], 'scaled'
                    elif mode == 'without_best_cell':
                        cc, method = [c for c in cols if features[c][1] != best], 'scaled'
                    else:
                        cc, method = cols, mode
                    pred = predict(tr, te, strains, cc, method)
                    for sample, pr in zip(strains, pred):
                        outputs.append(dict(window=window,date=date,worm_key=animal,sample_id=sample,
                                            mode=mode,prediction=pr,correct=int(sample==pr),
                                            chance=1/len(strains),n_features=len(cc),selected_cell=best))
    predictions = pd.DataFrame(outputs)
    predictions.to_csv(OUT/'tables/information_predictions.csv', index=False)
    pd.DataFrame(choices).to_csv(OUT/'tables/information_training_choices.csv', index=False)
    animal = predictions.groupby(['window','mode','date','worm_key'], as_index=False).agg(
        accuracy=('correct','mean'),chance=('chance','first'),n_stimuli=('sample_id','nunique'))
    animal.to_csv(OUT/'tables/information_animal_accuracy.csv',index=False)
    rng = np.random.default_rng(SEED)
    unique = animal[['date','worm_key']].drop_duplicates().sort_values(['date','worm_key'])
    keys = list(unique.itertuples(index=False,name=None))
    group_ix = [np.flatnonzero(unique.date.to_numpy()==d) for d in unique.date.unique()]
    # Shared bootstrap weights retain every outcome/model from the same animal.
    draws = np.zeros((2000,len(keys)),int)
    for b in range(len(draws)):
        for ix in group_ix:
            np.add.at(draws[b], rng.choice(ix,len(ix),replace=True), 1)
    summaries = []
    for (window,mode), d in animal.groupby(['window','mode']):
        v = d.set_index(['date','worm_key']).reindex(keys).accuracy.to_numpy()
        boot = draws @ v / len(v)
        lo, hi = np.quantile(boot,[.025,.975])
        summaries.append(dict(window=window,mode=mode,accuracy=v.mean(),ci_low=lo,ci_high=hi,
                              n_animals=len(v),n_animal_strain=int(d.n_stimuli.sum()),
                              chance=d.chance.mean(),min_block_accuracy=d.groupby('date').accuracy.mean().min()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'tables/information_summary.csv',index=False)
    paired = []
    comparisons = [('raw','magnitude'),('raw_shape','magnitude'),('scaled','scaled_magnitude'),
                   ('scaled','magnitude'),('scaled','best_cell'),('shape','magnitude'),
                   ('without_best_cell','magnitude'),('scaled','rank1')]
    for w in ['stim','post','both','full','pooled30']:
        sub = animal[animal.window.eq(w)].pivot(index=['date','worm_key'],columns='mode',values='accuracy').reindex(keys)
        for left,right in comparisons:
            delta = (sub[left]-sub[right]).to_numpy()
            lo,hi = np.quantile(draws@delta/len(delta),[.025,.975])
            paired.append(dict(window=w,comparison=f'{left} - {right}',delta=delta.mean(),ci_low=lo,ci_high=hi,
                               animals_improved=int((delta>0).sum()),animals_worse=int((delta<0).sum())))
    for m in ['raw','scaled','shape','raw_shape','best_cell']:
        sub = animal[animal['mode'].eq(m)].pivot(index=['date','worm_key'],columns='window',values='accuracy').reindex(keys)
        delta=(sub.both-sub.stim).to_numpy()
        lo,hi=np.quantile(draws@delta/len(delta),[.025,.975])
        paired.append(dict(window=m,comparison='both - stim',delta=delta.mean(),ci_low=lo,ci_high=hi,
                           animals_improved=int((delta>0).sum()),animals_worse=int((delta<0).sum())))
        delta=(sub.both-sub.full).to_numpy()
        lo,hi=np.quantile(draws@delta/len(delta),[.025,.975])
        paired.append(dict(window=m,comparison='both - full',delta=delta.mean(),ci_low=lo,ci_high=hi,
                           animals_improved=int((delta>0).sum()),animals_worse=int((delta<0).sum())))
        delta=(sub.both-sub.pooled30).to_numpy()
        lo,hi=np.quantile(draws@delta/len(delta),[.025,.975])
        paired.append(dict(window=m,comparison='both - pooled30',delta=delta.mean(),ci_low=lo,ci_high=hi,
                           animals_improved=int((delta>0).sum()),animals_worse=int((delta<0).sum())))
    pd.DataFrame(paired).to_csv(OUT/'tables/information_comparisons.csv',index=False)
    block=animal.groupby(['window','mode','date']).accuracy.mean().reset_index()
    block.to_csv(OUT/'tables/information_block_diagnostic.csv',index=False)
    (OUT/'logs/information_methods.json').write_text(json.dumps(dict(
        seed=SEED,bootstrap=2000,bootstrap_unit='whole animals within observed sampling blocks',
        evaluation='leave one animal out within each block, all available candidate strains',
        inference='conditional on blocks; shared fitted training sets mean bootstrap is descriptive only',
        features='available in every candidate test strain and >=2 training animals per candidate',
        selection='best single cell selected by inner animal CV; PCA rank1 and scaling fitted on training only',
        warning='same dataset as discovery; no independent date validation or confirmatory p values'),indent=2))
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))


if __name__ == '__main__':
    main()

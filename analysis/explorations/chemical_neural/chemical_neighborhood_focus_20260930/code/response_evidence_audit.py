"""Re-express existing held-animal paired contrasts; no new model or inference.

Run from the repository using .pixi/envs/default/bin/python <this-file>.
Inputs are immutable previous-round tables. Dates identify acquisition blocks and
date + worm_key identifies the biological replicate. A held-animal row is not an
independent replicate of another contrast measured in the same animal.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT / 'reports/population_first_20260930/tables'


def animal_equal(d, column='correct'):
    return float(d.groupby(['date', 'worm_key'])[column].mean().mean())


def counts(d):
    return dict(n_pairs=d.pair_id.nunique(), n_animals=len(d[['date','worm_key']].drop_duplicates()),
                n_blocks=d.date.nunique(), n_animal_pairs=len(d),
                n_strains=len(set(d.strain_a) | set(d.strain_b)))


def summary_row(d, subset, variant, mode):
    pair = d.groupby('pair_id').correct.agg(['mean', 'size'])
    return dict(subset=subset, variant=variant, mode=mode, **counts(d),
                animal_equal_same_direction=animal_equal(d),
                pair_equal_same_direction=float(pair['mean'].mean()),
                animal_pair_equal_same_direction=float(d.correct.mean()),
                animal_equal_mean_cosine=animal_equal(d, 'signed_cosine'),
                descriptive_animal_pair_median_cosine=float(d.signed_cosine.median()),
                pairs_more_than_half_same_direction=int(pair['mean'].gt(.5).sum()),
                pairs_half_same_direction=int(pair['mean'].eq(.5).sum()),
                pairs_less_than_half_same_direction=int(pair['mean'].lt(.5).sum()),
                pairs_all_same_direction=int(pair['mean'].eq(1).sum()),
                pairs_none_same_direction=int(pair['mean'].eq(0).sum()),
                min_animals_per_pair=int(pair['size'].min()),
                max_animals_per_pair=int(pair['size'].max()))


def directly_verify_vectors(nearest):
    """Independently rebuild all main / gain-removed stored contrast vectors."""
    wide=pd.read_parquet(SOURCE/'aligned_neural_animal_5bins.parquet').reset_index()
    wide['date']=wide.date.astype(str)
    wide['worm_key']=wide.worm_key.astype(str)
    columns=[c for c in wide if '__' in c]
    assert len(columns)==65
    errors=[]
    for date,data in wide.groupby('date'):
        animals=sorted(data.worm_key.unique()); strains=sorted(data.sample_id.unique())
        index=pd.MultiIndex.from_product([animals,strains],names=['worm_key','sample_id'])
        tensor=data.set_index(['worm_key','sample_id']).reindex(index)[columns].to_numpy().reshape(
            len(animals),len(strains),13,5)
        here=nearest[nearest.date.eq(date)&nearest.variant.eq('mean')&nearest['mode'].isin(['population','gain_removed'])]
        for row in here.itertuples():
            ai=animals.index(row.worm_key)
            ss=[strains.index(row.strain_a),strains.index(row.strain_b)]
            train=tensor[np.arange(len(animals))!=ai]
            paired=train[:,ss]
            test=tensor[ai,ss]
            complete=np.isfinite(paired).all(axis=(1,3))
            usable=np.isfinite(test).all(axis=(0,2)) & (complete.sum(axis=0)>=2)
            scale=np.maximum(np.nanstd(train[:,:,usable,:],axis=(0,1,3)),1e-6)
            mean=[]
            for cell in np.flatnonzero(usable):
                # Pair-complete training animals are the only animals entering
                # each cell. This handles cell missingness without filling.
                mean.append(paired[complete[:,cell],:,cell,:].mean(axis=0))
            template=np.stack(mean,axis=1)/scale[None,:,None]
            test=test[:,usable,:]/scale[None,:,None]
            template=template.reshape(2,-1);test=test.reshape(2,-1)
            if row.mode=='gain_removed':
                template=template/np.maximum(np.sqrt(np.mean(template**2,axis=1,keepdims=True)),1e-12)
                test=test/np.maximum(np.sqrt(np.mean(test**2,axis=1,keepdims=True)),1e-12)
            reference=template[0]-template[1];held=test[0]-test[1]
            dot=np.dot(reference,held)
            cosine=dot/max(np.linalg.norm(reference)*np.linalg.norm(held),1e-12)
            projection=dot/max(np.linalg.norm(reference),1e-12)
            assert usable.sum()==row.n_cells
            errors.append((abs(cosine-row.signed_cosine),abs(projection-row.signed_projection)))
    assert len(errors)==510
    max_error=np.max(errors,axis=0)
    assert max_error[0]<1e-12 and max_error[1]<1e-10
    return {'n_independent_vector_reconstructions':len(errors),
            'signed_cosine_max_absolute_error':float(max_error[0]),
            'signed_projection_max_absolute_error':float(max_error[1])}


def main():
    pred = pd.read_csv(SOURCE / 'neural_value_predictions.csv', dtype={'date':str,'worm_key':str})
    pairs = pd.read_csv(SOURCE / 'neural_value_pairs.csv', dtype={'date':str})
    old = pd.read_csv(SOURCE / 'neural_value_summary.csv')
    assert not pred.duplicated(['pair_id','date','worm_key','variant','mode']).any()
    same = np.where(pred.signed_projection > 0, 1., np.where(pred.signed_projection < 0, 0., .5))
    assert np.array_equal(same, pred.correct.to_numpy())
    assert np.array_equal(np.sign(pred.signed_projection),np.sign(pred.signed_cosine))
    assert np.isfinite(pred[['signed_cosine','signed_projection','correct']]).all().all()
    joined = pred.merge(pairs, on=['pair_id','date'], validate='many_to_one')
    nearest = joined[joined.nearest_either.eq(1)].copy()
    direct_check=directly_verify_vectors(nearest)
    nearest['same_direction'] = nearest.correct
    nearest.to_csv(OUT / 'tables/response_audit_animal_pair_directions.csv', index=False)

    # Explicit pair-level distributions keep inconsistent contrasts visible.
    pair_summary = nearest.groupby(['pair_id','variant','mode'], as_index=False).agg(
        n_animals=('worm_key','size'), same_direction_count=('correct','sum'),
        same_direction_fraction=('correct','mean'), median_cosine=('signed_cosine','median'),
        min_cosine=('signed_cosine','min'),max_cosine=('signed_cosine','max'),
        median_scaled_projection=('signed_projection','median'),
        min_cells=('n_cells','min'),max_cells=('n_cells','max'),
        min_paired_training_animals_per_cell=('n_train_animals','min'))
    pair_summary = pair_summary.merge(pairs, on='pair_id', validate='many_to_one')
    pair_summary['direction_support'] = np.select(
        [pair_summary.same_direction_fraction.eq(1),pair_summary.same_direction_fraction.gt(.5),
         pair_summary.same_direction_fraction.eq(.5)],
        ['all_measured_animals','majority_measured_animals','half_measured_animals'],
        default='less_than_half_measured_animals')
    pair_summary.to_csv(OUT / 'tables/response_audit_pair_directions.csv', index=False)

    masks = {'nearest_either':nearest.nearest_either.eq(1),
             'nearest_group_ge3':nearest.group_n_strains.ge(3),
             'same_species_nearest':nearest.same_species.eq(1),
             'mutual_nearest':nearest.mutual_nearest.eq(1)}
    summaries=[]
    animal_rows=[]
    for label, mask in masks.items():
        for (variant,mode),d in nearest[mask].groupby(['variant','mode']):
            summaries.append(summary_row(d,label,variant,mode))
            a=d.groupby(['date','worm_key'],as_index=False).agg(
                same_direction_fraction=('correct','mean'), n_pairs=('pair_id','nunique'),
                mean_cosine=('signed_cosine','mean'),median_cosine=('signed_cosine','median'))
            a['subset']=label;a['variant']=variant;a['mode']=mode
            animal_rows.append(a)
    summaries=pd.DataFrame(summaries)
    matched=summaries.merge(old, on=['subset','variant','mode'],validate='one_to_one')
    error=float(np.max(np.abs(matched.animal_equal_same_direction-matched.accuracy)))
    assert error < 1e-12
    summaries.to_csv(OUT / 'tables/response_audit_subset_summary.csv', index=False)
    pd.concat(animal_rows,ignore_index=True).to_csv(OUT / 'tables/response_audit_animal_summary.csv',index=False)

    # Challenge dominance with fixed stored vectors. These are descriptive
    # deletions, not retraining or confidence intervals. On each deletion the
    # remaining contrasts are averaged within animal, then animals equally.
    primary=nearest[nearest.variant.eq('mean') & nearest['mode'].eq('population')]
    sensitivity=[]
    def add(label,removed,d):
        sensitivity.append(dict(deletion_type=label,removed=removed,**counts(d),
            animal_equal_same_direction=animal_equal(d),
            animal_equal_mean_cosine=animal_equal(d,'signed_cosine'),
            pair_equal_same_direction=float(d.groupby('pair_id').correct.mean().mean())))
    add('none','none',primary)
    for pair in sorted(primary.pair_id.unique()):
        add('one_pair',pair,primary[primary.pair_id.ne(pair)])
    for strain in sorted(set(primary.strain_a)|set(primary.strain_b)):
        add('one_strain',strain,primary[primary.strain_a.ne(strain)&primary.strain_b.ne(strain)])
    for block in sorted(primary.date.unique()):
        add('one_block',block,primary[primary.date.ne(block)])
    strength=primary.groupby('pair_id').agg(fraction=('correct','mean'),cosine=('signed_cosine','median'))
    strength=strength.sort_values(['fraction','cosine'],ascending=False)
    for k in [3,5,10]:
        removed=list(strength.index[:k])
        add('highest_direction_fraction_then_cosine', ';'.join(removed),
            primary[~primary.pair_id.isin(removed)])
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / 'tables/response_audit_deletions.csv',index=False)

    # A shared strain can enter several neighbors; display that dependence.
    incidence=pd.concat([primary.rename(columns={'strain_a':'strain'}),
                         primary.rename(columns={'strain_b':'strain'})],ignore_index=True)
    incidence.groupby('strain',as_index=False).agg(
        n_pairs=('pair_id','nunique'),n_animal_pair_entries=('pair_id','size'),
        mean_direction_fraction=('correct','mean')).to_csv(
            OUT / 'tables/response_audit_strain_incidence.csv',index=False)
    for path in OUT.glob('tables/response_audit_*.csv'):
        assert path.stat().st_size > 0
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [SOURCE/'neural_value_predictions.csv',SOURCE/'neural_value_pairs.csv',
                        SOURCE/'neural_value_summary.csv',SOURCE/'aligned_neural_animal_5bins.parquet',Path(__file__)]}
    (OUT/'logs/response_evidence_verification.json').write_text(json.dumps(
        dict(status='success',method='fixed previous-round predictions; no refit',
             old_summary_max_absolute_error=error,
             sign_agrees_for_all_rows=True,n_source_prediction_rows=len(pred),
             **direct_check,
             source_and_script_sha256=manifest),indent=2)+'\n')
    print(summaries[summaries.variant.eq('mean')&summaries['mode'].isin(['population','gain_removed'])].to_string(index=False))
    print(sensitivity.groupby('deletion_type').animal_equal_same_direction.agg(['min','max']).to_string())


if __name__=='__main__':
    main()

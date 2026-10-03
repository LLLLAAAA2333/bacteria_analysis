"""Chemistry-selected neighbors and independent-animal population discrimination.

All neurons start on equal footing. Chemistry alone selects pairs; no neural
outcome enters pair selection. In each held-out animal, paired assignment asks
whether its A-minus-B population difference agrees with training animals. It is
not independent identification of unknown stimuli or new-date validation.
"""
from pathlib import Path
import json
import itertools
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OLD = ROOT/'reports/exploration_20260929/tables'
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
BINS = [(0,5),(5,10),(10,15),(15,20),(20,25)]
SEED = 2026093022


def build_features(curves):
    x = curves.reset_index()
    x['date'] = x['date'].astype(str)
    for j,(lo,hi) in enumerate(BINS):
        x[f'bin{j}'] = x[[str(i) for i in range(lo,hi)]].mean(axis=1)
    return x


def read_inputs():
    a = pd.read_parquet(OLD/'animal_curves.parquet')
    t = pd.read_parquet(OLD/'trial_curves.parquet').reset_index()
    t['date'] = t['date'].astype(str)
    ix = ['sample_id','date','worm_key','neuron_class']
    t = t.sort_values('segment_index')
    first = t.groupby(ix,as_index=False).first().set_index(ix)[a.columns]
    t['trial_rank'] = t.groupby(ix).cumcount()
    later = t[t.trial_rank.gt(0)].groupby(ix)[a.columns].mean()
    feature = {'mean':build_features(a),'first':build_features(first),'later':build_features(later)}
    tf = build_features(t.set_index(ix+['segment_index'])[a.columns])
    bins = [f'bin{j}' for j in range(5)]
    detrended = []
    for _,d in tf.groupby(['date','worm_key','neuron_class']):
        x = d.segment_index.to_numpy(float)
        x -= x.mean()
        y = d[bins].to_numpy(float)
        slope = (x[:,None]*y).sum(axis=0) / max(np.dot(x,x),1e-12)
        adjusted = d.copy()
        adjusted[bins] = y - x[:,None]*slope
        detrended.append(adjusted)
    feature['order_detrended'] = pd.concat(detrended).groupby(ix,as_index=False)[bins].mean()
    design = t[['sample_id','date','worm_key','segment_index']].drop_duplicates()
    design.groupby(['sample_id','date','worm_key'],as_index=False).segment_index.agg(['min','mean','max']).to_csv(OUT/'tables/neural_value_order_design.csv',index=False)
    chem_path = OUT/'tables/aligned_chemical_log2fc_paired.csv'
    # The independently audited alignment output is preferred. A legacy table
    # is numerically identical, but a provenance assertion below records this.
    if not chem_path.exists():
        chem_path = OLD/'chemical_legacy_logfc.csv'
    c = pd.read_csv(chem_path).set_index('sample_id')
    legacy = pd.read_csv(OLD/'chemical_legacy_logfc.csv').set_index('sample_id')
    assert c.shape == (106,380), c.shape
    assert set(c.columns)==set(legacy.columns)
    assert np.allclose(c.loc[legacy.index,legacy.columns],legacy,atol=1e-12,rtol=0)
    tax = pd.read_csv(OLD/'taxonomy.csv').set_index('sample_id')
    ref = pd.read_csv(OLD/'chemical_reference_groups.csv').set_index('sample_id')
    aligned = pd.read_parquet(OUT/'tables/aligned_neural_animal_5bins.parquet')
    own = feature['mean'].set_index(['sample_id','date','worm_key','neuron_class'])
    for neuron in NEURONS:
        ownpart = own.xs(neuron,level='neuron_class')[[f'bin{j}' for j in range(5)]]
        target = aligned.reindex(ownpart.index)[[f'{neuron}__{lo:02d}_{hi:02d}' for lo,hi in BINS]]
        assert np.allclose(ownpart.to_numpy(float), target.to_numpy(float), atol=1e-12,rtol=0,equal_nan=True)
    return a, feature, c, tax, ref, chem_path


def define_pairs(feature,chem,tax,ref):
    coverage = feature['mean'][['date','sample_id']].drop_duplicates()
    coverage['genus'] = coverage.sample_id.map(tax.genus_clean)
    coverage['reference'] = coverage.sample_id.map(ref.reference_group)
    pairs = []
    for (block,genus,reference),d in coverage.groupby(['date','genus','reference']):
        ids = sorted(d.sample_id)
        if len(ids)<2:
            continue
        distances = {(x,y):float(np.sqrt(np.mean((chem.loc[x]-chem.loc[y])**2)))
                     for x,y in itertools.combinations(ids,2)}
        nearest = {x:min((y for y in ids if y!=x),key=lambda y:(distances[tuple(sorted([x,y]))],y)) for x in ids}
        for (x,y),distance in distances.items():
            pairs.append(dict(pair_id=f'{block}_{x}_{y}',date=block,strain_a=x,strain_b=y,
                              genus=genus,reference_group=reference,chemical_rms_log2fc=distance,
                              group_n_strains=len(ids),nearest_either=int(nearest[x]==y or nearest[y]==x),
                              mutual_nearest=int(nearest[x]==y and nearest[y]==x),
                              species_a=tax.loc[x,'species_clean'].strip(),species_b=tax.loc[y,'species_clean'].strip()))
    pairs = pd.DataFrame(pairs).sort_values(['chemical_rms_log2fc','pair_id']).reset_index(drop=True)
    pairs['chemical_rank'] = np.arange(1,len(pairs)+1)
    pairs['closest_quartile'] = (pairs.chemical_rank <= int(np.ceil(len(pairs)/4))).astype(int)
    pairs['same_species'] = (pairs.species_a==pairs.species_b).astype(int)
    return pairs


def tensors(feature):
    result = {}
    for block,d in feature['mean'].groupby('date'):
        animals = sorted(d.worm_key.unique())
        strains = sorted(d.sample_id.unique())
        tensors = {}
        for variant,v in feature.items():
            z = v[v.date.eq(block)].set_index(['worm_key','sample_id','neuron_class'])
            arr = np.full((len(animals),len(strains),len(NEURONS),len(BINS)),np.nan)
            for ai,animal in enumerate(animals):
                for si,strain in enumerate(strains):
                    for ni,neuron in enumerate(NEURONS):
                        if (animal,strain,neuron) in z.index:
                            arr[ai,si,ni,:] = z.loc[(animal,strain,neuron),[f'bin{j}' for j in range(len(BINS))]].to_numpy(float)
            tensors[variant] = arr
        result[block] = (animals,strains,tensors)
    return result


def evaluate(pairs,blocks):
    rows = []
    for pair in pairs.itertuples():
        animals,strains,arrs = blocks[pair.date]
        ss = [strains.index(pair.strain_a),strains.index(pair.strain_b)]
        common = np.isfinite(arrs['mean']).all(axis=(0,1,3))
        for ai,animal in enumerate(animals):
            others = np.array([i for i in range(len(animals)) if i!=ai])
            for variant,trvar,tevar in [('mean','mean','mean'),('first_to_later','first','later'),('later_to_first','later','first'),('order_detrended','order_detrended','order_detrended')]:
                pool = arrs[trvar][others]
                tr = pool[:,ss]
                te = arrs[tevar][ai,ss]
                # One shared set of observed cells for A and B, no identity-
                # specific imputation. >=2 complete paired training animals/cell.
                present = np.isfinite(tr).all(axis=(1,3))
                available = np.isfinite(te).all(axis=(0,2)) & (present.sum(axis=0)>=2)
                for mode in ['population','gain_removed','magnitude_only','common_cells','without_largest_cell','largest_cell_only']:
                    mask = available & common if mode=='common_cells' else available
                    if mask.sum()<3:
                        continue
                    # One scale per neuron, estimated only from training animals
                    # across their stimuli and the same five windows. No label
                    # selection, cell selection, or tuned hyperparameters.
                    sd = np.nanstd(pool[:,:,mask,:],axis=(0,1,3))
                    sd = np.maximum(sd,1e-6)
                    trsub = tr[:,:,mask,:] / sd[None,None,:,None]
                    tesub = te[:,mask,:] / sd[None,:,None]
                    # Coordinate-wise paired templates use only animals with
                    # both strains at that neuron. This preserves all available
                    # cells without selecting only fully imaged animals.
                    paired = np.isfinite(trsub).all(axis=1)
                    trsub = np.where(paired[:,None,:,:],trsub,np.nan)
                    centroid = np.nanmean(trsub,axis=0).reshape(2,-1)
                    tesub = tesub.reshape(2,-1)
                    ntrain = paired.all(axis=2).sum(axis=0)
                    selected_cells = list(np.array(NEURONS)[mask])
                    cell_difference = (centroid[0]-centroid[1]).reshape(-1,5)
                    biggest_i = int(np.argmax(np.sum(cell_difference**2,axis=1)))
                    largest_cell = selected_cells[biggest_i]
                    if mode in ['without_largest_cell','largest_cell_only']:
                        keep_cells = np.arange(len(selected_cells))!=biggest_i if mode=='without_largest_cell' else np.arange(len(selected_cells))==biggest_i
                        centroid = centroid.reshape(2,-1,5)[:,keep_cells,:].reshape(2,-1)
                        tesub = tesub.reshape(2,-1,5)[:,keep_cells,:].reshape(2,-1)
                        selected_cells = list(np.array(selected_cells)[keep_cells])
                    if mode=='gain_removed':
                        centroid /= np.maximum(np.sqrt(np.mean(centroid**2,axis=1,keepdims=True)),1e-12)
                        tesub /= np.maximum(np.sqrt(np.mean(tesub**2,axis=1,keepdims=True)),1e-12)
                    elif mode=='magnitude_only':
                        centroid = np.sqrt(np.mean(centroid**2,axis=1,keepdims=True))
                        tesub = np.sqrt(np.mean(tesub**2,axis=1,keepdims=True))
                    direction = centroid[0]-centroid[1]
                    difference = tesub[0]-tesub[1]
                    dot = float(np.dot(direction,difference))
                    cosine = dot / max(float(np.linalg.norm(direction)*np.linalg.norm(difference)),1e-12)
                    rows.append(dict(pair_id=pair.pair_id,date=pair.date,worm_key=animal,variant=variant,mode=mode,
                                     n_cells=len(selected_cells),cells=';'.join(selected_cells),largest_training_cell=largest_cell,
                                     n_train_animals=int(ntrain.min()),max_train_animals=int(ntrain.max()),correct=float((dot>0)+.5*(dot==0)),
                                     signed_cosine=cosine,signed_projection=dot/max(np.linalg.norm(direction),1e-12)))
    return pd.DataFrame(rows)


def summarize(pred,pairs):
    z = pred.merge(pairs,on=['pair_id','date'],validate='many_to_one')
    pair_summary = z.groupby(['pair_id','variant','mode'],as_index=False).agg(
        accuracy=('correct','mean'),n_animals=('worm_key','nunique'),median_cosine=('signed_cosine','median'),
        min_cells=('n_cells','min'),max_cells=('n_cells','max'),min_train_animals=('n_train_animals','min'))
    pair_summary = pair_summary.merge(pairs,on='pair_id',validate='many_to_one')
    subset_masks = {'all_comparable':np.ones(len(z),bool),'nearest_either':z.nearest_either.eq(1),
                    'mutual_nearest':z.mutual_nearest.eq(1),'closest_quartile':z.closest_quartile.eq(1),
                    'same_species_nearest':z.nearest_either.eq(1)&z.same_species.eq(1),
                    'nearest_group_ge3':z.nearest_either.eq(1)&z.group_n_strains.ge(3)}
    animal_rows = []
    for label,mask in subset_masks.items():
        d = z[mask].groupby(['variant','mode','date','worm_key'],as_index=False).agg(
            accuracy=('correct','mean'),n_pairs=('pair_id','nunique'),median_cosine=('signed_cosine','median'))
        d['subset'] = label
        animal_rows.append(d)
    animal = pd.concat(animal_rows,ignore_index=True)
    rng = np.random.default_rng(SEED)
    summary=[]
    for (subset,variant,mode),d in animal.groupby(['subset','variant','mode']):
        d = d.reset_index(drop=True)
        draws = np.zeros((2000,len(d)),int)
        for _,ix in d.groupby('date').groups.items():
            ix=np.asarray(ix)
            sampled=rng.choice(ix,size=(2000,len(ix)),replace=True)
            for i in range(2000):
                np.add.at(draws[i],sampled[i],1)
        lo,hi = np.quantile(draws @ d.accuracy.to_numpy()/len(d),[.025,.975])
        m = z[(z.variant.eq(variant))&(z['mode'].eq(mode))]
        if subset!='all_comparable':
            if subset=='same_species_nearest': m=m[m.nearest_either.eq(1)&m.same_species.eq(1)]
            elif subset=='nearest_group_ge3': m=m[m.nearest_either.eq(1)&m.group_n_strains.ge(3)]
            else: m=m[m[subset].eq(1)]
        summary.append(dict(subset=subset,variant=variant,mode=mode,accuracy=d.accuracy.mean(),
                            descriptive_lo=lo,descriptive_hi=hi,n_animals=len(d),n_blocks=d.date.nunique(),
                            n_pairs=m.pair_id.nunique(),n_strains=len(set(m.strain_a)|set(m.strain_b)),
                            n_animal_pairs=len(m),animal_above_chance=int(d.accuracy.gt(.5).sum()),
                            animal_below_chance=int(d.accuracy.lt(.5).sum())))
    comparisons = []
    for (subset,variant),d in animal.groupby(['subset','variant']):
        w = d.pivot(index=['date','worm_key'],columns='mode',values='accuracy').dropna()
        for left,right in [('population','magnitude_only'),('gain_removed','magnitude_only'),('common_cells','population'),('without_largest_cell','population'),('without_largest_cell','largest_cell_only')]:
            delta = (w[left]-w[right]).to_numpy()
            draws = []
            indexes = np.arange(len(w))
            dates = w.index.get_level_values('date')
            for b in range(2000):
                sampled = np.concatenate([rng.choice(indexes[dates==date],sum(dates==date),replace=True) for date in dates.unique()])
                draws.append(delta[sampled].mean())
            lo,hi = np.quantile(draws,[.025,.975])
            comparisons.append(dict(subset=subset,variant=variant,comparison=f'{left} - {right}',
                                    delta=delta.mean(),descriptive_lo=lo,descriptive_hi=hi,n_animals=len(w),
                                    improved=int((delta>0).sum()),worse=int((delta<0).sum()),tied=int((delta==0).sum())))
    pd.DataFrame(comparisons).to_csv(OUT/'tables/neural_value_comparisons.csv',index=False)
    return pair_summary,animal,pd.DataFrame(summary)


def plot_examples(curves,pairs,pred,tax):
    colors=['#247796','#c57035']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    c=curves.reset_index(); c['date']=c.date.astype(str)
    times=np.arange(-5,25)
    exported=[]
    for rank in [1,2]:
        pair=pairs[pairs.chemical_rank.eq(rank)].iloc[0]
        fig,axes=plt.subplots(4,4,figsize=(11.2,9.6),sharex=True,layout='constrained')
        for ni,neuron in enumerate(NEURONS):
            ax=axes.flat[ni]
            for si,strain in enumerate([pair.strain_a,pair.strain_b]):
                d=c[c.date.eq(pair.date)&c.sample_id.eq(strain)&c.neuron_class.eq(neuron)]
                values=d[[str(t) for t in times]].to_numpy()
                for rr,(animal,y) in enumerate(zip(d.worm_key,values)):
                    ax.plot(times,y,color=colors[si],alpha=.22,lw=.65)
                    exported.extend(dict(pair_id=pair.pair_id,chemical_rank=rank,sample_id=strain,neuron_class=neuron,
                                         date=pair.date,worm_key=animal,time=int(t),response=float(v)) for t,v in zip(times,y))
                if len(values):ax.plot(times,values.mean(axis=0),color=colors[si],lw=1.8,label=f'{strain}, n={len(values)}')
            ax.axhline(0,color='.7',lw=.55);ax.axvspan(0,10,color='.88',zorder=-1)
            ax.set_title(neuron,loc='left',fontsize=10)
            ax.set_xlim(-5,24);ax.set_xticks([0,10,20])
            ax.legend(frameon=False,fontsize=7,loc='best')
            if ni%4==0:ax.set_ylabel('Calcium ΔF/F₀')
            if ni>=9:
                ax.set_xlabel('Time from stimulus onset (s)')
                ax.tick_params(labelbottom=True)
        for j in [13,14,15]:axes.flat[j].axis('off')
        main=pred[pred.pair_id.eq(pair.pair_id)&pred.variant.eq('mean')&pred['mode'].eq('population')]
        shape=pred[pred.pair_id.eq(pair.pair_id)&pred.variant.eq('mean')&pred['mode'].eq('gain_removed')]
        axes.flat[13].text(0,1,f'Chemistry-only choice #{rank}\n{pair.strain_a} vs {pair.strain_b}\n{pair.species_a}\n{pair.species_b}\n\nLC–MS RMS log₂FC distance: {pair.chemical_rms_log2fc:.2f}\nSame genus and reference group',va='top',fontsize=9)
        axes.flat[14].text(0,1,f'Independent-animal paired assignment\nPopulation: {main.correct.sum():g}/{len(main)} correct\nGain removed: {shape.correct.sum():g}/{len(shape)} correct\n\nThin curves: individual animals\nThick curves: animal mean\nGray: stimulus present\nEach neuron has its own y scale',va='top',fontsize=9)
        axes.flat[15].text(0,1,'Pairs were ranked using all 380\nchemical features, before viewing\nneural responses.\n\nAll 13 neurons are shown.\nFive 5-s bins from 0 to 25 s\nwere used for assignment.\nMetadata are in source tables.',va='top',fontsize=9)
        fig.suptitle(f'What population responses reveal about a chemical near-neighbor pair: {pair.strain_a} / {pair.strain_b}',fontsize=12)
        stem=OUT/f'figures/neural_value_chemical_neighbor_{rank}'
        fig.savefig(stem.with_suffix('.png'),dpi=180,bbox_inches='tight');fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    pd.DataFrame(exported).to_csv(OUT/'tables/neural_value_example_curves.csv',index=False)
    paired_rows = []
    fig,axes = plt.subplots(1,2,figsize=(9,5.4),sharey=True,layout='constrained')
    for rank,ax in zip([1,2],axes):
        pair=pairs[pairs.chemical_rank.eq(rank)].iloc[0]
        for ni,neuron in enumerate(NEURONS):
            d=c[c.date.eq(pair.date)&c.sample_id.isin([pair.strain_a,pair.strain_b])&c.neuron_class.eq(neuron)].copy()
            d['response_mean']=d[[str(t) for t in range(25)]].mean(axis=1)
            w=d.pivot(index='worm_key',columns='sample_id',values='response_mean').dropna()
            delta=w[pair.strain_a]-w[pair.strain_b]
            ys=ni+np.linspace(-.16,.16,len(delta))
            ax.scatter(delta,ys,s=24,color='#247796',alpha=.75,edgecolor='white',linewidth=.4)
            ax.scatter([delta.mean()],[ni],s=38,color='black',marker='|',linewidth=1.6)
            for animal,v in delta.items():
                paired_rows.append(dict(pair_id=pair.pair_id,chemical_rank=rank,date=pair.date,worm_key=animal,
                                        neuron_class=neuron,response_difference_mean_0_25=float(v)))
        ax.axvline(0,color='.5',lw=.8)
        ax.set_yticks(range(13),NEURONS);ax.set_ylim(12.6,-.6)
        ax.set_xlabel(f'{pair.strain_a} − {pair.strain_b}\nMean calcium ΔF/F₀, 0–25 s')
        nper=c[c.date.eq(pair.date)&c.sample_id.eq(pair.strain_a)].groupby('neuron_class').worm_key.nunique()
        ax.set_title(f'Chemical neighbor #{rank}: {pair.strain_a} / {pair.strain_b}\nRMS log₂FC = {pair.chemical_rms_log2fc:.2f}; {nper.min()}–{nper.max()} animals/cell',fontsize=10)
        ax.grid(axis='y',color='.93',lw=.6)
    # Shared x scale is required for direct comparison of effect size.
    bound=max(abs(v['response_difference_mean_0_25']) for v in paired_rows)*1.08
    for ax in axes:ax.set_xlim(-bound,bound)
    fig.suptitle('Population response differences within individual animals\nEach point is one animal; black ticks are paired-animal means',fontsize=11)
    fig.savefig(OUT/'figures/neural_value_paired_population.png',dpi=200,bbox_inches='tight')
    fig.savefig(OUT/'figures/neural_value_paired_population.pdf',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(paired_rows).to_csv(OUT/'tables/neural_value_example_paired_differences.csv',index=False)


def export_distribution_and_verify(pred,pairs,summary):
    z=pred[pred.variant.eq('mean')&pred['mode'].eq('population')].merge(pairs[['pair_id','nearest_either']]).query('nearest_either==1')
    count=z.groupby('largest_training_cell').agg(n_animal_pair_folds=('pair_id','size'),n_unique_pairs_ever_selected=('pair_id','nunique')).reindex(NEURONS,fill_value=0)
    modal=z.groupby('pair_id').largest_training_cell.agg(lambda x: sorted(x.value_counts()[x.value_counts().eq(x.value_counts().max())].index))
    count['n_pairs_modal_including_ties']=[sum(n in x for x in modal) for n in count.index]
    count.index.name='neuron_class'
    count.reset_index().to_csv(OUT/'tables/neural_value_largest_cell_distribution.csv',index=False)
    assert int(count.n_animal_pair_folds.sum())==len(z)
    assert pred.correct.isin([0,.5,1]).all()
    assert pred.n_train_animals.ge(2).all()
    assert pred.n_cells.ge(1).all()
    assert pairs.chemical_rms_log2fc.gt(0).all()
    animal = z.groupby(['date','worm_key']).correct.mean()
    reported = summary[(summary.subset=='nearest_either')&(summary.variant=='mean')&(summary['mode']=='population')].accuracy.iloc[0]
    assert np.isclose(animal.mean(),reported,atol=1e-14)
    keys = ['pair_id','date','worm_key','variant','mode']
    assert not pred.duplicated(keys).any()
    for variant,d in pred.groupby('variant'):
        keys_by_mode = [set(x[['pair_id','date','worm_key']].itertuples(index=False,name=None)) for _,x in d.groupby('mode')]
        assert all(k==keys_by_mode[0] for k in keys_by_mode)
    check = dict(status='passed',chemical_dimensions=[106,380],neural_bins=BINS,
                 exact_fc_and_neural_alignment='asserted in read_inputs; tolerance 1e-12',
                 paired_nearest_pairs=int(z.pair_id.nunique()),animal_pair_folds=len(z),
                 animal_weighted_accuracy=float(animal.mean()),all_modes_matched_observations=True,
                 source_inputs_unchanged='read-only source paths; root manifest checks source hashes',
                 figures='all three manually rendered and inspected; two examples chemistry-rank #1 and #2',
                 generalization='new animals conditional on the sampled protocol, not independent acquisition validation')
    (OUT/'logs/neural_value_verification.json').write_text(json.dumps(check,indent=2))


def main():
    curves,features,chem,tax,ref,chem_path=read_inputs()
    pairs=define_pairs(features,chem,tax,ref)
    blocks=tensors(features)
    pred=evaluate(pairs,blocks)
    ps,animal,summary=summarize(pred,pairs)
    for name,df in [('pairs',pairs),('predictions',pred),('pair_summary',ps),('animal_summary',animal),('summary',summary)]:
        df.to_csv(OUT/f'tables/neural_value_{name}.csv',index=False)
    export_distribution_and_verify(pred,pairs,summary)
    plot_examples(curves,pairs,pred,tax)
    methods=dict(seed=SEED,chemical_input=str(chem_path.relative_to(ROOT)),chemical_sha256=hashlib.sha256(chem_path.read_bytes()).hexdigest(),
                 chemistry='106 strains x 380 exact log2FC, no QC removal or z scaling; RMS Euclidean distance',
                 bins=BINS,unit='One held-out animal per within-block strain pair; trials averaged, cells not replicate units',
                 pair_selection='All same-block same-genus same-reference pairs; chemistry-only nearest either, mutual nearest and closest quartile',
                 primary='nearest-either paired assignment using all available neurons, scaled once/cell using only training animals',
                 paired_assignment='Positive dot product of training mean A-B and held-out A-B; 50% chance; identities both known present',
                 training='At least two paired training animals per cell; coordinate-wise templates then global normalization; no neural feature/pair selection',
                 gain_removed='Normalize each mean training template and each test vector about physical zero after training-only cell scaling; retains signs and relative amplitudes',
                 largest_cell_check='Largest training-only squared difference summed over all five bins after cell scaling; remove it or use it alone, no held-out outcomes in selection',
                 uncertainty='2000 block-stratified whole-animal bootstrap draws of fixed predictions; descriptive, not refit or independent validation',
                 limitations=['All trials/stimuli of a test animal excluded from scale and template fitting',
                              'Sampling block and arbitrary stimulus-order carryover remain; no new-date generalization',
                              'Order sensitivity removes only a linear trial-position trend per animal/cell/bin across all stimuli, not specific preceding-stimulus effects',
                              'Pairs share strains/animals; pair points are not independent repetitions',
                              'Single chemical measurement has no replicate error estimate; no claim of intrinsic neural superiority',
                              'Chemical neighbors are relative neighbors; distance 0 would be identical, these distances are not 0'])
    (OUT/'logs/neural_value_methods.json').write_text(json.dumps(methods,indent=2))
    print(summary.round(4).to_string(index=False))
    print(ps[ps.chemical_rank.le(2)].round(4).to_string(index=False))

if __name__=='__main__':main()

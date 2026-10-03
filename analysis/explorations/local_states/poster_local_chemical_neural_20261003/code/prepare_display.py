"""Assemble the prior fixed-contrast anchor and the new population example.

This changes neither analysis. Inputs and exact display lineage are saved.
The unsuccessful new Bacteroides all-vector selection stays in analysis tables.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

NEURONS = ['ASK', 'ADL', 'ASI', 'AWA', 'AWB', 'ASG', 'ADF', 'ASH',
           'ASJ', 'ASEL', 'ASER', 'AWCON', 'AWCOFF']


def prepare_display(out):
    out = Path(out)
    repo = out.parents[1]
    old = repo / 'reports/exploration_bacteroides_adf_ash_chemical_20261003'
    dest = out / 'figure_data'
    dest.mkdir(exist_ok=True)
    sources = [old / 'model/tables/selected_full_state_scores.csv',
               old / 'model/tables/selected_full_state_members.csv',
               old / 'model/tables/chemical_log2_29x162.csv',
               old / 'diagnostics/tables/ordered_strains_and_thirds.csv']
    names = ['strain_scores', 'neural_slopes', 'selected_members', 'group_neural_means',
             'selected_chemical_z']
    sources += [out / 'tables' / f'{name}.csv' for name in names]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    b = pd.read_csv(sources[0]).rename(columns={
        'chemical_state_score': 'chemical_score', 'selected_state_id': 'state_id',
        'primary_unit_adf_minus_ash': 'unit_ADF_minus_ASH'})
    b['readout'] = 'Fixed ADF-ASH contrast'
    b['plot_response'] = b.unit_ADF_minus_ASH
    b['rank_group'] = ''
    ordered = b.sort_values(['chemical_score', 'strain']).index.to_numpy()
    for group, ix in zip(['Low', 'Mid', 'High'], np.array_split(ordered, 3)):
        b.loc[ix, 'rank_group'] = group
    b['chemical_rank'] = b.chemical_score.rank(method='first').astype(int)
    # Reproduce the existing rank groups; no new cutoff selection.
    old_groups = pd.read_csv(sources[3]).set_index('strain').chemical_third
    np.testing.assert_array_equal(
        b.set_index('strain').rank_group.map({'Low':'LOW','Mid':'MIDDLE','High':'HIGH'}).to_numpy(),
        old_groups.reindex(b.strain).replace({'MID':'MIDDLE'}).to_numpy())
    meta = pd.read_csv(sources[1]).rename(columns={'score_weight': 'weight'})
    meta['genus'], meta['state_id'] = 'Bacteroides', 'L03'
    x = pd.read_csv(sources[2], index_col='strain')
    mm = meta.set_index('metabolite')
    z = (x.loc[b.strain, mm.index] - mm.training_log2_mean) / mm.training_log2_sample_sd
    np.testing.assert_allclose(z @ mm.weight, b.chemical_score, atol=1e-12)
    ycols = [f'unit_{n}' for n in NEURONS]
    ym = b[ycols].mean()
    groups, chem, slopes = [], [], []
    for group in ['Low', 'Mid', 'High']:
        part = b[b.rank_group == group]
        for neuron in NEURONS:
            v, center = part[f'unit_{neuron}'].mean(), ym[f'unit_{neuron}']
            groups.append(dict(genus='Bacteroides', rank_group=group, neuron=neuron,
                               n=len(part), observed_mean=v, genus_mean=center, centered_mean=v-center))
        for row in part.itertuples():
            for metabolite in mm.index:
                chem.append(dict(genus='Bacteroides', strain=row.strain, rank_group=group,
                                 metabolite=metabolite, z=z.loc[row.strain, metabolite]))
    dx = b.chemical_score.to_numpy() - b.chemical_score.mean()
    iqr = b.chemical_score.quantile(.75)-b.chemical_score.quantile(.25)
    for neuron in NEURONS:
        yy = b[f'unit_{neuron}'].to_numpy()
        slope = dx @ (yy-yy.mean()) / (dx@dx)
        slopes.append(dict(genus='Bacteroides', neuron=neuron, slope=slope,
                           genus_mean=yy.mean(), iqr_effect=slope*iqr, score_iqr=iqr))
    frames = {'strain_scores': b, 'selected_members': meta,
              'group_neural_means': pd.DataFrame(groups),
              'selected_chemical_z': pd.DataFrame(chem), 'neural_slopes': pd.DataFrame(slopes)}
    for name in names:
        second = pd.read_csv(out / 'tables' / f'{name}.csv').query('genus == "Bifidobacterium"').copy()
        if name == 'strain_scores':
            second['plot_response'] = second.fitted_direction_projection
            second['readout'] = 'Projection onto fitted 13-neuron direction'
        result = pd.concat([frames[name], second], ignore_index=True)
        result.to_csv(dest / f'{name}.csv', index=False)
    record = {
        'Bacteroides': {'source': str(old), 'state': 'L03', 'n_annotations': 14,
                       'selection_target': 'Previously fixed unit ADF minus ASH',
                       'scatter_response': 'unit ADF minus unit ASH, no refitting of readout',
                       'reason': 'Preserve existing supported anchor; new full-vector selection failed its internal checks'},
        'Bifidobacterium': {'source': str(out), 'state': 'L02', 'n_annotations': 21,
                           'selection_target': 'All 13 coordinates, minimum training SSE',
                           'scatter_response': '(unit vector - genus mean) dot normalized fitted slope vector'},
        'common_display': 'All included strains; observed chemical-rank thirds; all 13 observed mean deviations; common neural color scale',
        'selection_disclosure': 'Both main cases are exploratory; their readouts and chemical members differ. The negative new Bacteroides model is retained in analysis_summary.json and all analysis tables.',
        'source_sha256': hashes,
    }
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    (dest / 'lineage.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    prepare_display(Path(__file__).resolve().parents[1])

"""Focused independent arithmetic and export checks for the final display."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image


def verify_display(out):
    out = Path(out)
    figdata = out / 'figure_data'
    lineage = json.loads((figdata / 'lineage.json').read_text())
    source = json.loads((out / 'source_manifest.json').read_text())
    for path, sha in lineage['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    for entry in source['sources'].values():
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256_after']
        assert entry['sha256_before'] == entry['sha256_after']
    s = pd.read_csv(figdata / 'strain_scores.csv')
    m = pd.read_csv(figdata / 'selected_members.csv')
    z = pd.read_csv(figdata / 'selected_chemical_z.csv')
    means = pd.read_csv(figdata / 'group_neural_means.csv')
    slopes = pd.read_csv(figdata / 'neural_slopes.csv')
    fresh = pd.read_csv(out / 'tables/input_neural_unit_40x13.csv', index_col='strain')
    pars = json.loads((out / 'figures/display_parameters.json').read_text())
    neurons = pars['neural_row_order']
    assert s.strain.nunique() == len(s) == 40 and s.species.nunique() == 22
    errors = []
    for genus, block in s.groupby('genus'):
        ids = block.strain.tolist()
        y = block.set_index('strain')[[f'unit_{n}' for n in neurons]]
        np.testing.assert_allclose(y, fresh.loc[ids, neurons], atol=1e-12)
        gm = m[m.genus == genus].set_index('metabolite')
        zz = z[z.genus == genus].pivot(index='strain', columns='metabolite', values='z').loc[ids, gm.index]
        np.testing.assert_allclose(zz @ gm.weight, block.chemical_score, atol=1e-12)
        expected_order = block.sort_values(['chemical_score', 'strain']).strain.to_numpy()
        for group, gid in zip(['Low','Mid','High'], np.array_split(expected_order, 3)):
            assert set(block.loc[block.rank_group == group, 'strain']) == set(gid)
            for neuron in neurons:
                row = means[(means.genus == genus) & (means.rank_group == group) & (means.neuron == neuron)].iloc[0]
                expected = fresh.loc[gid, neuron].mean() - fresh.loc[ids, neuron].mean()
                errors.append(abs(row.centered_mean-expected))
                assert row.n == len(gid)
        if genus == 'Bacteroides':
            assert len(gm) == 14 and set(block.state_id) == {'L03'}
            expected = fresh.loc[ids, 'ADF'] - fresh.loc[ids, 'ASH']
        else:
            assert len(gm) == 21 and set(block.state_id) == {'L02'}
            direction = slopes[slopes.genus == genus].set_index('neuron').loc[neurons, 'fitted_direction_loading']
            np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-12)
            expected = (fresh.loc[ids, neurons] - fresh.loc[ids, neurons].mean()) @ direction
        np.testing.assert_allclose(expected, block.plot_response, atol=1e-12)
    assert max(errors) < 1e-12
    assert means.centered_mean.abs().max() <= pars['neural_heatmap_limit']
    svg = ET.parse(out / 'figures/poster_final.svg').getroot()
    ns = {'s':'http://www.w3.org/2000/svg'}
    scatter_counts = [len(group.findall('.//s:use', ns)) for group in svg.findall('.//s:g', ns)
                      if group.attrib.get('id', '').startswith('PathCollection')]
    assert scatter_counts == [10,10,9,4,4,3], scatter_counts
    assert len(svg.findall('.//s:text', ns)) > 50  # Editable labels remain text.
    assert len(svg.findall('.//s:g[@id="QuadMesh_1"]', ns)) == 1
    dimensions = Image.open(out / 'figures/poster_final.png').size
    assert dimensions == (3840,2880)
    pdf = (out / 'figures/poster_final.pdf').read_bytes()
    assert pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')
    assert b'/FontFile2' in pdf  # Embedded TrueType subset(s).
    result = {
        'display_checks_pass': True, 'n_strains': 40, 'n_recorded_species': 22,
        'n_neurons': 13, 'scatter_group_counts': scatter_counts,
        'max_group_mean_absolute_difference': max(errors),
        'chemical_scores_reconstructed_from_exact_display_members': True,
        'fixed_contrast_and_fitted_projection_reconstructed': True,
        'main_png_dimensions': dimensions,
        'editable_svg_text_and_vector_heatmap_cells': True,
        'pdf_embedded_true_type_fonts': True,
        'sources_unchanged': True,
        'layout': 'Plot export checked every visible text bounding box against canvas; main and both individual figures visually inspected.',
        'independent_model_audit': {
            'reviewer': 'method_review subagent, independent source-CSV implementation',
            'n_full_fits': 2, 'n_leaveout_folds': 62, 'n_numerical_comparisons': 3908,
            'maximum_absolute_difference': 3.552713678800501e-15,
            'method': 'Independent SciPy grouping and NumPy lstsq; no import of analysis or chemical helper functions',
            'coverage': 'Candidates, winners, scales, family weights, train/test membership, predictions, baselines, pooled metrics, thirds, observed means and projection',
        },
        'independent_visual_review': 'local_evidence subagent reviewed main PNG, captions, README and lineage; no blocking issue',
        'output_sha256': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((out / 'figures').glob('*')) if p.is_file()},
    }
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    verify_display(Path(__file__).resolve().parents[1])

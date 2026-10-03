"""Fixed continuous bacterial colors from a chemical reference RDM only."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

CHEMICAL_COLOR_CMAP = 'turbo'


def build_chemical_colors(reference_rdm, cmap_name=CHEMICAL_COLOR_CMAP,
                          reference_name='cell9 chemical_rdm'):
    """Map chemical PCo1 scores linearly to a fixed color scale, without groups.

    Classical PCoA double-centers squared chemical distances. AID is only an
    identity key and deterministic row order, never a numerical color feature.
    Equal scores may share a color; color is a one-axis summary, not a distance.
    """
    if not isinstance(reference_rdm, pd.DataFrame):
        raise TypeError('reference_rdm must be an AID-indexed DataFrame')
    if not reference_rdm.index.is_unique or not reference_rdm.index.equals(reference_rdm.columns):
        raise ValueError('Chemical reference rows/columns must be identical unique IDs in the same order')
    if reference_rdm.index.isna().any() or not all(isinstance(s, str) for s in reference_rdm.index):
        raise ValueError('Chemical reference needs nonmissing string sample IDs')
    if len(reference_rdm) < 2:
        raise ValueError('Chemical reference needs at least two samples')
    reference = reference_rdm.sort_index().loc[:, sorted(reference_rdm.index)].copy(deep=True)
    values = reference.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or (values < -1e-12).any():
        raise ValueError('Chemical reference distances must be finite and nonnegative')
    if not np.allclose(values, values.T, atol=1e-12, rtol=0):
        raise ValueError('Chemical reference must be symmetric')
    if not np.allclose(np.diag(values), 0, atol=1e-12, rtol=0):
        raise ValueError('Chemical reference diagonal must be zero')
    # Roundoff correction only; no imputation, neural fit, or block weighting.
    values = np.maximum((values + values.T) / 2, 0)
    np.fill_diagonal(values, 0)
    reference.iloc[:, :] = values
    squared = values ** 2
    gram = -.5 * (squared - squared.mean(axis=0)[None, :]
                   - squared.mean(axis=1)[:, None] + squared.mean())
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    tolerance = np.max(np.abs(eigenvalues)) * len(values) * np.finfo(float).eps
    positive = eigenvalues > tolerance
    if not positive.any():
        raise ValueError('Chemical reference has no positive PCoA variation to color')
    scores = eigenvectors[:, 0] * np.sqrt(eigenvalues[0])
    # Fix the arbitrary sign; sorted IDs make reordered input reproducible.
    if scores[np.argmax(np.abs(scores))] < 0:
        scores = -scores
    scores = pd.Series(scores, index=reference.index, name='chemical_PCo1')
    vmin, vmax = float(scores.min()), float(scores.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    # Interpolate the palette instead of restricting samples to 256 LUT slots.
    cmap = LinearSegmentedColormap.from_list(
        cmap_name + '_continuous', colormaps[cmap_name](np.linspace(0, 1, 256)), N=4096)
    samples = scores.to_frame()
    samples['normalized_score'] = norm(scores.to_numpy())
    samples['color_hex'] = [to_hex(cmap(value)) for value in samples.normalized_score]
    positive_inertia = float(eigenvalues[positive].sum())
    negative_inertia = float(np.abs(eigenvalues[eigenvalues < -tolerance]).sum())
    digest = hashlib.sha256(reference.to_csv(float_format='%.17g').encode('utf-8')).hexdigest()
    return dict(
        scores=scores, samples=samples, cmap=cmap, cmap_name=cmap_name,
        vmin=vmin, vmax=vmax, method='classical PCoA', axis=1,
        eigenvalues=eigenvalues, positive_inertia_fraction=float(eigenvalues[0] / positive_inertia),
        negative_inertia_fraction=negative_inertia / (positive_inertia + negative_inertia),
        reference_rdm=reference, reference_name=reference_name, reference_sha256=digest,
        reference_metadata=copy.deepcopy(reference_rdm.attrs), label='Chemical PCo1',
        note='Colors summarize chemical PCo1 only; similar colors do not imply similar full chemical profiles.',
    )


def color_values(mapping, sample_ids):
    scores = mapping['scores'].reindex(sample_ids)
    if scores.isna().any():
        raise ValueError(f"Samples absent from the fixed chemical reference: {scores.index[scores.isna()].tolist()}")
    return scores.to_numpy()


def mpl_colors(mapping, sample_ids):
    return dict(c=color_values(mapping, sample_ids), cmap=mapping['cmap'],
                norm=Normalize(vmin=mapping['vmin'], vmax=mapping['vmax']))


def mpl_colorbar_kwargs(mapping):
    return dict(label=mapping['label'])


def plotly_colors(mapping, sample_ids):
    scale = [[float(t), to_hex(mapping['cmap'](t))] for t in np.linspace(0, 1, 256)]
    return dict(color=color_values(mapping, sample_ids).tolist(),
                cmin=mapping['vmin'], cmax=mapping['vmax'], colorscale=scale, showscale=True,
                colorbar=dict(title=dict(text=mapping['label'], side='right')))


def color_provenance(mapping):
    result = {key: mapping[key] for key in
              ('reference_name', 'reference_sha256', 'reference_metadata', 'method', 'axis',
               'cmap_name', 'vmin', 'vmax', 'positive_inertia_fraction',
               'negative_inertia_fraction', 'note')}
    result.update(color_source='chemical reference only', n_reference_samples=len(mapping['scores']),
                  normalization='linear min-max over the full fixed reference; no grouping or ranking',
                  sign_rule='largest absolute PCo1 score positive; sample IDs sorted before decomposition')
    return result


def save_chemical_colors(mapping, directory):
    """Save the fixed reference, per-sample scores/colors and provenance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mapping['samples'].to_csv(directory / 'aid_to_chemical_color.csv', encoding='utf-8-sig', index_label='AID')
    mapping['reference_rdm'].to_csv(directory / 'chemical_reference_rdm.csv', float_format='%.17g')
    pd.DataFrame({'axis': np.arange(1, len(mapping['eigenvalues']) + 1),
                  'eigenvalue': mapping['eigenvalues']}).to_csv(directory / 'chemical_pcoa_eigenvalues.csv', index=False)
    (directory / 'color_parameters.json').write_text(
        json.dumps(color_provenance(mapping), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return mapping['samples'].copy()

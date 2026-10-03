"""Centered neural PCA and sign/rotation-invariant top-two comparisons."""
import numpy as np


def fit_pca(values):
    """Mean-center without coordinate scaling; largest-absolute loading positive."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError('PCA needs a finite matrix with at least three rows')
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])
    signs[signs == 0] = 1
    loadings = (vt * signs[:, None]).T
    variance = singular ** 2 / (len(values) - 1)
    return {'mean': mean, 'centered': centered, 'loadings': loadings,
            'scores': centered @ loadings, 'singular_values': singular,
            'variance': variance, 'ratio': variance / variance.sum()}


def compare_subspaces(reference_loadings, other_loadings):
    """Compare fixed top-two spans plus the single PC1 direction.

    Inputs must be orthonormal loading matrices with >=2 columns. Projector
    distance is Frobenius/sqrt(2), range [0,sqrt(2)], not a normalized angle.
    """
    v = np.asarray(reference_loadings, float)[:, :2]
    u = np.asarray(other_loadings, float)[:, :2]
    if v.shape != u.shape or v.shape[1] != 2:
        raise ValueError('Both loading bases must have matching d x 2 top-two columns')
    np.testing.assert_allclose(v.T @ v, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=1e-10)
    cosines = np.clip(np.linalg.svd(u.T @ v, compute_uv=False), 0, 1)
    angles = np.degrees(np.arccos(cosines))
    pc1_cosine = float(np.clip(abs(u[:, 0] @ v[:, 0]), 0, 1))
    projector_distance = float(np.linalg.norm(u @ u.T - v @ v.T, ord='fro') / np.sqrt(2))
    return {'principal_angle_1_deg': float(angles[0]), 'principal_angle_2_deg': float(angles[1]),
            'max_principal_angle_deg': float(angles.max()), 'projector_frobenius_over_sqrt2': projector_distance,
            'pc1_absolute_cosine': pc1_cosine, 'pc1_acute_angle_deg': float(np.degrees(np.arccos(pc1_cosine)))}


def spectrum_metrics(fit):
    eigenvalues = fit['variance']
    ratios = fit['ratio']
    return {'pc1_evr': float(ratios[0]), 'pc2_evr': float(ratios[1]), 'top2_evr': float(ratios[:2].sum()),
            'lambda1': float(eigenvalues[0]), 'lambda2': float(eigenvalues[1]), 'lambda3': float(eigenvalues[2]),
            'lambda1_minus_lambda2': float(eigenvalues[0] - eigenvalues[1]),
            'lambda2_minus_lambda3': float(eigenvalues[1] - eigenvalues[2]),
            'relative_gap_2_vs_3': float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[1]),
            'lambda2_over_lambda3': float(eigenvalues[1] / eigenvalues[2])}

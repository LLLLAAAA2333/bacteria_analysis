"""Pure, fold-local chemical axis discovery and transformation.

Every fit uses only rows in log_frame. IDs are local to each fit. No files,
classification labels, neural values or prior module definitions are read.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

ZERO_SD = 1e-12
CUT_DISTANCE = 0.5


def _validate_log_frame(log_frame):
    if not isinstance(log_frame, pd.DataFrame):
        raise TypeError("log_frame must be a pandas DataFrame")
    if not log_frame.index.is_unique or not log_frame.columns.is_unique:
        raise ValueError("strain index and feature columns must be unique")
    if not np.isfinite(log_frame.to_numpy(dtype=float)).all():
        raise ValueError("log_frame must contain only finite numeric values")


def _align_metadata(feature_metadata, features):
    metadata = feature_metadata.copy()
    if "metabolite" in metadata.columns:
        metadata = metadata.set_index("metabolite")
    if not metadata.index.is_unique:
        raise ValueError("feature_metadata must have unique metabolite IDs")
    if "family" not in metadata.columns:
        raise ValueError("feature_metadata requires a family column")
    missing = [f for f in features if f not in metadata.index]
    if missing:
        raise ValueError(f"Missing metadata for features: {missing}")
    metadata = metadata.loc[features]
    if metadata.family.isna().any() or metadata.family.astype(str).str.strip().eq("").any():
        raise ValueError("Every input feature needs a nonempty family identifier")
    return metadata


def fit_axes(log_frame, feature_metadata):
    """Return a dict of chemical axes learned only from log_frame training rows.

    Required input: >=2 rows of finite log2 concentration values. Zero-SD features
    are retained in means/scales but omitted from grouping. No eligible modules
    returns an n x 0 train_scores frame without inventing another axis type.
    """
    _validate_log_frame(log_frame)
    if len(log_frame) < 2:
        raise ValueError("fit_axes requires at least two training strains")
    features = log_frame.columns.tolist()
    metadata = _align_metadata(feature_metadata, features)
    means = log_frame.mean(axis=0)
    scales = log_frame.std(axis=0, ddof=1)
    retained = scales.index[scales > ZERO_SD].tolist()
    excluded = scales.index[scales <= ZERO_SD].tolist()
    z = (log_frame.loc[:, retained] - means.loc[retained]) / scales.loc[retained]
    groups = []
    feature_order = retained.copy()
    if len(retained) >= 2:
        correlations = np.corrcoef(z.to_numpy().T)
        distances = np.clip(1 - correlations, 0, 2)
        np.fill_diagonal(distances, 0)
        tree = linkage(squareform(distances, checks=False), method="average", optimal_ordering=True)
        labels = fcluster(tree, t=CUT_DISTANCE, criterion="distance")
        feature_order = [retained[i] for i in leaves_list(tree)]
        for label in sorted(set(labels)):
            members = [retained[i] for i in np.flatnonzero(labels == label)]
            if len(members) >= 3 and metadata.loc[members, "family"].nunique() >= 3:
                groups.append(members)
    groups.sort(key=lambda members: (-len(members), min(members)))
    module_members = {f"L{i:02d}": members for i, members in enumerate(groups, 1)}
    score_weights = {}
    for module, members in module_members.items():
        family = metadata.loc[members, "family"]
        counts = family.value_counts()
        score_weights[module] = pd.Series(
            [1.0 / (len(counts) * counts[f]) for f in family], index=members, name="score_weight"
        )
    fitted = {
        "means": means.copy(), "scales": scales.copy(),
        "source_features": features, "retained_features": retained, "excluded_features": excluded,
        "train_ids": log_frame.index.tolist(), "module_members": module_members,
        "score_weights": score_weights, "feature_order": feature_order,
        "parameters": {"scale_ddof": 1, "zero_sd": ZERO_SD, "correlation": "Pearson",
                       "distance": "1-r", "linkage": "average", "cut_distance": CUT_DISTANCE,
                       "min_members": 3, "min_families": 3,
                       "score_weighting": "equal family, then equal member"},
    }
    fitted["train_scores"] = transform_axes(log_frame, fitted)
    return fitted


def transform_axes(log_frame, fitted):
    """Apply training means/scales/members without refitting; preserve row order."""
    _validate_log_frame(log_frame)
    missing = [f for f in fitted["source_features"] if f not in log_frame.columns]
    if missing:
        raise ValueError(f"Missing fitted source features: {missing}")
    modules = fitted["module_members"]
    output = pd.DataFrame(index=log_frame.index, columns=list(modules), dtype=float)
    for module, members in modules.items():
        z = (log_frame.loc[:, members] - fitted["means"].loc[members]) / fitted["scales"].loc[members]
        output[module] = z @ fitted["score_weights"][module].loc[members]
    return output

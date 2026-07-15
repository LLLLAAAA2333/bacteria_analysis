"""Minimal preprocessing helpers — derive trial_id, baseline-center, reshape to tensor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER

TRACE_GROUP_COLUMNS = ("trial_id", "stimulus", "neuron")


def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a normalized trial_id column (date + worm + segment)."""

    out = df.copy()
    date = pd.to_datetime(out["date"], errors="raise").dt.strftime("%Y%m%d")
    out["trial_id"] = date + "__" + out["worm_key"].astype(str) + "__" + out["segment_index"].astype(str)
    return out


def center_by_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Center traces by the mean of baseline time points (t0-t5)."""

    out = df if "trial_id" in df.columns else add_trial_id(df)

    stats = (
        out.groupby(list(TRACE_GROUP_COLUMNS), sort=False, dropna=False)
        .apply(lambda g: g.loc[g["time_point"].isin(BASELINE_TIMEPOINTS), "delta_F_over_F0"].mean(), include_groups=False)
        .reset_index(name="baseline_mean")
    )
    out = out.merge(stats, on=list(TRACE_GROUP_COLUMNS), how="left")
    out["dff_baseline_centered"] = out["delta_F_over_F0"] - out["baseline_mean"]
    return out


def build_trial_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build one metadata row per trial, sorted by date."""

    source = df if "trial_id" in df.columns else add_trial_id(df)

    rows = []
    for trial_id, trial in source.groupby("trial_id", sort=False, dropna=False):
        first = trial.iloc[0]
        observed = {n for n in trial["neuron"].dropna().astype(str) if n in NEURON_ORDER}
        rows.append({
            "trial_id": trial_id,
            "date": first["date"],
            "worm_key": first["worm_key"],
            "segment_index": first["segment_index"],
            "stimulus": first["stimulus"],
            "stim_name": first["stim_name"],
            "stim_color": first["stim_color"],
            "n_observed_neurons": len(observed),
            "n_missing_neurons": len(NEURON_ORDER) - len(observed),
        })

    metadata = pd.DataFrame(rows)
    metadata["__sort_date"] = pd.to_datetime(metadata["date"], errors="raise")
    metadata = metadata.sort_values(
        ["__sort_date", "worm_key", "segment_index", "trial_id"], kind="stable"
    ).drop(columns=["__sort_date"]).reset_index(drop=True)
    return metadata


def build_trial_tensor(df: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
    """Build trial-major 3D array (trial × neuron × time) ordered by metadata."""

    source = df if "trial_id" in df.columns else add_trial_id(df)

    tensor = np.full((len(metadata), len(NEURON_ORDER), len(EXPECTED_TIMEPOINTS)), np.nan, dtype=float)
    trial_pos = {tid: i for i, tid in enumerate(metadata["trial_id"])}
    neuron_pos = {n: i for i, n in enumerate(NEURON_ORDER)}
    time_pos = {t: i for i, t in enumerate(EXPECTED_TIMEPOINTS)}

    for row in source.itertuples(index=False):
        ti = trial_pos.get(row.trial_id)
        ni = neuron_pos.get(row.neuron)
        tpi = time_pos.get(row.time_point)
        if ti is not None and ni is not None and tpi is not None:
            tensor[ti, ni, tpi] = row.dff_baseline_centered

    return tensor

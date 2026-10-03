"""Prepare animal responses using the final A–D Notebook aggregation order.

All response values retain the input ``delta_F_over_F0`` units and baseline.
One volume is one second; stimulus onset/offset are source indices 5/15.
The returned observations cover [0, 40) s relative to stimulus onset.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CLASSES = (
    "ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
    "ASEL", "ASER", "AWCON", "AWCOFF",
)
BILATERAL = {
    "ASK": ("ASKL", "ASKR"), "ADL": ("ADLL", "ADLR"),
    "ASI": ("ASIL", "ASIR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"), "ASG": ("ASGL", "ASGR"),
    "ADF": ("ADFL", "ADFR"), "ASH": ("ASHL", "ASHR"),
    "ASJ": ("ASJL", "ASJR"),
}
EXCLUDED_NEURONS = {"AFDL", "AFDR", "BAGL", "BAGR", "CEPR", "URXR"}
ONSET, OFFSET, BIN_SECONDS, N_BINS = 5, 15, 5, 8


def _prepare_table(source):
    """Aggregate a raw table; separated from file I/O for focused checks.

    Means skip NaN, including a missing partner or trial. Entirely missing
    groups remain NaN. Crucially, trials are averaged *per volume*, before
    volumes are averaged within each 5 s bin, as in the source Notebook.
    """
    source = source.copy()
    identity = ["date", "worm_key", "segment_index", "stim_name", "neuron"]
    if source[identity].isna().any().any():
        raise ValueError("Missing raw identity fields; cannot assign animal/trial/stimulus")
    if not source["start_time"].eq(ONSET).all() or not source["end_time"].eq(OFFSET).all():
        raise ValueError("Raw stimulus markers differ from onset=5 and exclusive offset=15")
    for column in ["date", "worm_key", "stim_name", "neuron"]:
        source[column] = source[column].astype(str)
    source["sample_id"] = source["stim_name"].str.extract(r"^(A\d{3})", expand=False)
    unknown = sorted(source.loc[source["sample_id"].isna(), "stim_name"].unique())
    if unknown:
        raise ValueError(f"Unrecognized stimulus names (not silently excluded): {unknown}")

    stimulus_mapping = source[["date", "stim_name", "sample_id"]].drop_duplicates()
    stimulus_mapping["suffix"] = stimulus_mapping["stim_name"].str.slice(4)
    stimulus_mapping = stimulus_mapping.sort_values(["date", "sample_id", "stim_name"])
    collisions = stimulus_mapping.groupby(["date", "sample_id"]).size()
    if (collisions > 1).any():
        raise ValueError(
            "Multiple stimulus names within date × strain require an explicit block mapping: "
            f"{collisions[collisions > 1].index.tolist()}"
        )
    unknown_neurons = sorted(set(source["neuron"]) - EXCLUDED_NEURONS - set(CLASSES)
                             - {n for pair in BILATERAL.values() for n in pair})
    if unknown_neurons:
        raise ValueError(f"Unexpected neuron labels: {unknown_neurons}")

    audit = {
        "n_input_rows_loaded": len(source),
        "n_input_animals": len(source[["date", "worm_key"]].drop_duplicates()),
        "n_input_trials": len(source[["date", "worm_key", "segment_index"]].drop_duplicates()),
        "n_input_nonfinite_responses": int((~np.isfinite(source["delta_F_over_F0"])).sum()),
        "excluded_neuron_rows": source.loc[source["neuron"].isin(EXCLUDED_NEURONS), "neuron"]
                                    .value_counts().sort_index().to_dict(),
        "unrecognized_stimulus_names": unknown,
        "unexpected_neurons": unknown_neurons,
        "stimulus_suffixes": sorted(stimulus_mapping["suffix"].unique().tolist()),
        "n_within_date_strain_name_collisions": int((collisions > 1).sum()),
    }
    source = source.loc[
        source["time_point"].between(ONSET, ONSET + BIN_SECONDS * N_BINS - 1)
        & ~source["neuron"].isin(EXCLUDED_NEURONS)
    ].copy()
    if source.empty:
        raise ValueError("No retained neuron observations in the requested 0–40 s window")
    neuron_map = {neuron: cell for cell, pair in BILATERAL.items() for neuron in pair}
    source["neuron_class"] = source["neuron"].map(lambda n: neuron_map.get(n, n))
    source["delta_F_over_F0"] = source["delta_F_over_F0"].replace([np.inf, -np.inf], np.nan)

    # These keys and this order reproduce final_load_responses (cell 28).
    trial_keys = ["sample_id", "date", "worm_key", "segment_index", "neuron_class", "time_point"]
    trials = source.groupby(trial_keys, as_index=False, observed=True)["delta_F_over_F0"].mean()
    trial_coverage = trials.groupby(trial_keys[:-1], as_index=False, observed=True).agg(
        n_observed_volumes=("delta_F_over_F0", "count"),
        n_recorded_volumes=("time_point", "size"),
    )
    trial_coverage["block"] = trial_coverage["date"]
    trial_coverage["animal_id"] = trial_coverage["date"] + "|" + trial_coverage["worm_key"]
    animal_keys = [key for key in trial_keys if key != "segment_index"]
    animals = trials.groupby(animal_keys, as_index=False, observed=True)["delta_F_over_F0"].mean()
    animals["bin_index"] = ((animals["time_point"] - ONSET) // BIN_SECONDS).astype(int)
    trace_keys = ["sample_id", "date", "worm_key", "neuron_class", "bin_index"]
    binned = animals.groupby(trace_keys, as_index=False, observed=True).agg(
        response=("delta_F_over_F0", "mean"), n_volumes=("delta_F_over_F0", "count"),
    )
    # Explicitly retain missing cells/windows for existing animal-condition pairs.
    observed_pairs = source[["sample_id", "date", "worm_key"]].drop_duplicates()
    features = pd.MultiIndex.from_product([CLASSES, range(N_BINS)],
                                         names=["neuron_class", "bin_index"]).to_frame(index=False)
    observations = observed_pairs.merge(features, how="cross").merge(binned, on=trace_keys, how="left")
    observations["n_volumes"] = observations["n_volumes"].fillna(0).astype(int)
    observations["block"] = observations["date"]
    observations["animal_id"] = observations["date"] + "|" + observations["worm_key"]
    observations["time_start"] = observations["bin_index"] * BIN_SECONDS
    observations["time_end"] = observations["time_start"] + BIN_SECONDS
    # Stable scientific cell order independent of lexical/category ordering.
    observations["neuron_class"] = pd.Categorical(observations["neuron_class"], CLASSES, ordered=True)
    observations = observations.sort_values(trace_keys).reset_index(drop=True)
    observations["neuron_class"] = observations["neuron_class"].astype(str)
    observations = observations[[
        "sample_id", "date", "block", "worm_key", "animal_id", "neuron_class",
        "bin_index", "time_start", "time_end", "response", "n_volumes",
    ]]

    condition_keys = ["sample_id", "date", "block", "neuron_class", "bin_index", "time_start", "time_end"]
    coverage = observations.groupby(condition_keys, as_index=False, observed=True).agg(
        n_animals=("response", "count"),
        n_recorded_animals=("animal_id", "size"),
        n_animals_full_bin=("n_volumes", lambda v: int(v.eq(BIN_SECONDS).sum())),
    )
    # Conditions with only excluded neurons still belong in the coverage audit.
    conditions = stimulus_mapping[["sample_id", "date"]].drop_duplicates()
    complete = conditions.merge(features, how="cross")
    complete["block"] = complete["date"]
    complete["time_start"] = complete["bin_index"] * BIN_SECONDS
    complete["time_end"] = complete["time_start"] + BIN_SECONDS
    coverage = complete.merge(coverage, on=condition_keys, how="left")
    counts = ["n_animals", "n_recorded_animals", "n_animals_full_bin"]
    coverage[counts] = coverage[counts].fillna(0).astype(int)

    audit.update({
        "n_retained_source_rows": len(source), "n_observation_rows": len(observations),
        "n_finite_observations": int(observations["response"].notna().sum()),
        "n_zero_observations": int(observations["response"].eq(0).sum()),
        "n_partial_bins": int(observations["n_volumes"].between(1, 4).sum()),
        "n_animals": observations["animal_id"].nunique(),
        "n_strains": stimulus_mapping["sample_id"].nunique(),
        "n_dates": stimulus_mapping["date"].nunique(),
        "dates": sorted(stimulus_mapping["date"].unique().tolist()),
        "n_conditions": len(conditions), "n_classes": len(CLASSES), "n_bins": N_BINS,
        "n_animal_conditions": len(observed_pairs),
        "n_zero_coverage_features": int(coverage["n_animals"].eq(0).sum()),
        "n_features_with_at_least_three_animals": int(coverage["n_animals"].ge(3).sum()),
    })
    return {"observations": observations, "coverage": coverage,
            "trial_coverage": trial_coverage.reset_index(drop=True),
            "stimulus_mapping": stimulus_mapping.reset_index(drop=True), "metadata": audit}


def prepare_responses(path):
    """Read a Parquet input without changing it; return data frames and metadata.

    ``block=date`` is the available recorded proxy, not independently verified
    culture/acquisition batch identity. Full stimulus names are audited before
    extracting strain IDs. Ambiguous within-date naming raises an error.
    """
    path = Path(path).resolve()
    columns = ["date", "worm_key", "segment_index", "stim_name", "neuron", "time_point",
               "delta_F_over_F0", "start_time", "end_time"]
    source = pd.read_parquet(path, columns=columns, engine="pyarrow",
                             read_dictionary=["date", "worm_key", "stim_name", "neuron"])
    prepared = _prepare_table(source)
    with path.open("rb") as handle:
        raw_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    prepared["metadata"].update({
        "raw_file": str(path), "raw_sha256": raw_sha256,
        "raw_size_bytes": path.stat().st_size,
        "raw_parquet_rows": pq.ParquetFile(path).metadata.num_rows,
        "source_logic": "02_reproducibility_inspection.ipynb final A–D cells 27–28",
        "aggregation": "L/R within trial and volume; trials within animal and volume; five volumes per bin",
        "classes": list(CLASSES), "bilateral_pooling": BILATERAL,
        "excluded_neurons": sorted(EXCLUDED_NEURONS),
        "animal_identity": ["date", "worm_key"], "block_definition": "date",
        "block_limitation": "Date is a proxy; an unrecorded within-date culture/acquisition block cannot be ruled out.",
        "onset_index": ONSET, "offset_index_exclusive": OFFSET,
        "volume_interval_seconds": 1, "window_seconds": [0, 40], "n_bins": N_BINS,
        "bin_seconds": BIN_SECONDS, "source_time_points": list(range(ONSET, ONSET + N_BINS * BIN_SECONDS)),
        "response_units": "delta_F_over_F0; unchanged input baseline, no extra centering or curve normalization",
        "missingness": "inf converted to NaN; available-value means at every aggregation; all-missing remains NaN",
        "n_volumes_definition": "Count of finite animal-level volume means in the five-second bin",
    })
    return prepared

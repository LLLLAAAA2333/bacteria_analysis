import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.constants import NEURON_ORDER
from bacteria_analysis import reliability as reliability_module


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise AssertionError(f"missing expected column; tried {candidates}, got {list(frame.columns)}")


def _pairwise_with_view_name(view_name: str, view: pd.DataFrame) -> pd.DataFrame:
    compute_pairwise_distances = getattr(reliability_module, "compute_pairwise_distances", None)
    assert compute_pairwise_distances is not None, "compute_pairwise_distances is not implemented yet"

    pairwise = compute_pairwise_distances(view)
    if "view_name" not in pairwise.columns:
        pairwise = pairwise.assign(view_name=view_name)
    return pairwise


def _correlation_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.allclose(left, right):
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    return float(1.0 - corr)


def test_view_windows_and_individual_id_construction(stage1_trial_metadata):
    assert reliability_module.VIEW_WINDOWS == {
        "full_trajectory": tuple(range(45)),
        "on_window": (6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        "response_window": tuple(range(6, 21)),
        "post_window": tuple(range(16, 45)),
    }
    assert reliability_module.DEFAULT_DISTANCE_METRIC == "correlation"

    assert reliability_module.build_individual_id("2026-03-27", "worm_001") == "2026-03-27__worm_001"

    annotated = reliability_module.add_individual_id(stage1_trial_metadata)
    assert "individual_id" in annotated.columns
    assert annotated["individual_id"].nunique() == 4
    assert annotated.loc[annotated["trial_id"] == "20260327__worm_001__0", "individual_id"].iloc[0] == "2026-03-27__worm_001"
    assert annotated.loc[annotated["trial_id"] == "20260328__worm_002__2", "individual_id"].iloc[0] == "2026-03-28__worm_002"


def test_build_trial_views_slices_expected_time_windows(stage1_trial_metadata, stage1_trial_tensor):
    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)

    assert set(views) == {"full_trajectory", "on_window", "response_window", "post_window"}
    assert all(hasattr(view, "values") for view in views.values())

    for view_name, expected_timepoints in reliability_module.VIEW_WINDOWS.items():
        view = views[view_name]

        assert tuple(view.timepoints) == expected_timepoints
        assert view.values.shape == (len(stage1_trial_metadata), len(NEURON_ORDER), len(expected_timepoints))

    full_view = views["full_trajectory"]
    trial_index = full_view.metadata.index[full_view.metadata["trial_id"] == "20260327__worm_001__0"][0]
    assert full_view.values[trial_index, NEURON_ORDER.index("ADFL"), 6] == pytest.approx(0.2)
    assert full_view.values[trial_index, NEURON_ORDER.index("ADFL"), 15] == pytest.approx(1.1)

    on_view = views["on_window"]
    assert on_view.timepoints == tuple(range(6, 16))

    response_view = views["response_window"]
    assert response_view.timepoints == tuple(range(6, 21))

    post_view = views["post_window"]
    assert post_view.timepoints == tuple(range(16, 45))


def test_overlap_aware_distance_uses_shared_neurons_only(stage1_trial_metadata, stage1_trial_tensor):
    compute_pairwise_distances = getattr(reliability_module, "compute_pairwise_distances", None)
    assert compute_pairwise_distances is not None, "compute_pairwise_distances is not implemented yet"

    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    full_view = views["full_trajectory"]
    pairwise = compute_pairwise_distances(full_view)

    distance_col = _pick_column(pairwise, ("distance", "pairwise_distance", "correlation_distance"))
    overlap_col = _pick_column(pairwise, ("n_overlap_neurons", "overlap_neuron_count"))
    left_trial_col = _pick_column(pairwise, ("trial_id_left", "trial_id_a", "trial_id_1"))
    right_trial_col = _pick_column(pairwise, ("trial_id_right", "trial_id_b", "trial_id_2"))

    b2_trial = "20260327__worm_001__1"
    b3_trial = "20260327__worm_001__2"
    match = pairwise[
        ((pairwise[left_trial_col] == b2_trial) & (pairwise[right_trial_col] == b3_trial))
        | ((pairwise[left_trial_col] == b3_trial) & (pairwise[right_trial_col] == b2_trial))
    ].iloc[0]

    assert match[overlap_col] == 1

    b2_index = int(np.where(full_view.metadata["trial_id"].astype(str) == b2_trial)[0][0])
    b3_index = int(np.where(full_view.metadata["trial_id"].astype(str) == b3_trial)[0][0])
    overlap_neuron = "ASEL"
    neuron_index = NEURON_ORDER.index(overlap_neuron)
    expected = _correlation_distance(full_view.values[b2_index, neuron_index, :], full_view.values[b3_index, neuron_index, :])

    assert match[distance_col] == pytest.approx(expected)


def test_same_vs_different_summary_is_separated_and_positive(stage1_trial_metadata, stage1_trial_tensor):
    build_trial_views = getattr(reliability_module, "build_trial_views")
    summarize_same_vs_different = getattr(reliability_module, "summarize_same_vs_different", None)
    assert summarize_same_vs_different is not None, "summarize_same_vs_different is not implemented yet"

    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    pairwise_frames = []
    for view_name, view in views.items():
        pairwise_frames.append(_pairwise_with_view_name(view_name, view))

    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    summary = summarize_same_vs_different(pairwise)

    view_col = _pick_column(summary, ("view_name", "representation_view"))
    same_col = _pick_column(summary, ("same_mean_distance", "mean_same_distance", "same_mean"))
    different_col = _pick_column(summary, ("different_mean_distance", "mean_different_distance", "different_mean"))
    gap_col = _pick_column(summary, ("distance_gap", "mean_distance_gap", "different_minus_same"))

    summary = summary.set_index(view_col)
    assert set(summary.index) == set(reliability_module.VIEW_WINDOWS)
    for view_name in reliability_module.VIEW_WINDOWS:
        assert summary.loc[view_name, same_col] < summary.loc[view_name, different_col]
        assert summary.loc[view_name, gap_col] > 0


def test_overlap_neuron_count_requires_shared_valid_timepoints():
    left = np.array([[1.0, np.nan, np.nan]])
    right = np.array([[np.nan, np.nan, 2.0]])

    comparison = reliability_module.compare_trial_arrays(left, right)

    assert comparison["overlap_neuron_count"] == 0
    assert comparison["comparison_status"] == "insufficient_overlap_neurons"


def test_leave_one_individual_out_isolated_by_individual_and_scores_above_chance(stage1_trial_metadata, stage1_trial_tensor):
    run_leave_one_group_out = getattr(reliability_module, "run_leave_one_group_out", None)
    assert run_leave_one_group_out is not None, "run_leave_one_group_out is not implemented yet"

    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    trial_frame, group_frame, summary_frame = run_leave_one_group_out(
        views["full_trajectory"],
        "individual_id",
        "individual",
    )

    held_out_col = _pick_column(group_frame, ("heldout_group", "held_out_individual_id", "held_out_group", "held_out_value"))
    score_col = _pick_column(group_frame, ("accuracy", "score", "reliability_score"))
    train_group_col = _pick_column(group_frame, ("n_training_trials", "n_train_individuals", "n_train_groups", "n_training_groups"))
    test_count_col = _pick_column(group_frame, ("n_heldout_trials", "n_test_trials", "n_held_out_trials"))

    assert held_out_col in group_frame.columns
    assert group_frame[held_out_col].nunique() == 4
    assert (group_frame[train_group_col] == 9).all()
    assert (group_frame[test_count_col] == 3).all()
    assert (group_frame[score_col] > 0.5).all()
    assert not trial_frame.empty
    assert not summary_frame.empty


def test_leave_one_date_out_isolated_by_date_and_scores_above_chance(stage1_trial_metadata, stage1_trial_tensor):
    run_leave_one_group_out = getattr(reliability_module, "run_leave_one_group_out", None)
    assert run_leave_one_group_out is not None, "run_leave_one_group_out is not implemented yet"

    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    trial_frame, group_frame, summary_frame = run_leave_one_group_out(
        views["full_trajectory"],
        "date",
        "date",
    )

    held_out_col = _pick_column(group_frame, ("heldout_group", "held_out_date", "held_out_group", "held_out_value"))
    score_col = _pick_column(group_frame, ("accuracy", "score", "reliability_score"))
    train_group_col = _pick_column(group_frame, ("n_training_trials", "n_train_dates", "n_train_groups", "n_training_groups"))
    test_count_col = _pick_column(group_frame, ("n_heldout_trials", "n_test_trials", "n_held_out_trials"))

    assert group_frame[held_out_col].nunique() == 2
    assert (group_frame[train_group_col] == 6).all()
    assert (group_frame[test_count_col] == 6).all()
    assert (group_frame[score_col] > 0.5).all()
    assert not trial_frame.empty
    assert not summary_frame.empty


def test_within_date_cross_individual_same_vs_different_is_positive(stage1_trial_metadata, stage1_trial_tensor):
    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    pairwise = pd.concat(
        [reliability_module.compute_pairwise_distances(view) for view in views.values()],
        ignore_index=True,
    )

    filtered, summary = reliability_module.summarize_within_date_cross_individual_same_vs_different(pairwise)

    assert not filtered.empty
    assert filtered["same_date"].all()
    assert (~filtered["same_individual"]).all()

    summary = summary.set_index("view_name")
    for view_name in reliability_module.VIEW_WINDOWS:
        assert summary.loc[view_name, "distance_gap"] > 0


def test_per_date_loio_runs_within_each_date(stage1_trial_metadata, stage1_trial_tensor):
    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)

    trial_frame, group_frame, summary_frame = reliability_module.run_per_date_loio(views["full_trajectory"])

    assert set(group_frame["source_date"]) == {"2026-03-27", "2026-03-28"}
    assert (group_frame["n_training_trials"] == 3).all()
    assert (group_frame["n_heldout_trials"] == 3).all()
    assert (group_frame["accuracy"] > 0.5).all()
    assert set(summary_frame["source_date"]) == {"2026-03-27", "2026-03-28"}
    assert not trial_frame.empty


def test_stimulus_distance_pairs_build_symmetric_matrix(stage1_trial_metadata, stage1_trial_tensor):
    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    pairwise = pd.concat(
        [reliability_module.compute_pairwise_distances(view) for view in views.values()],
        ignore_index=True,
    )

    pair_summary = reliability_module.summarize_stimulus_distance_pairs(pairwise)

    response_pairs = pair_summary.loc[pair_summary["view_name"] == "response_window"].copy()
    assert not response_pairs.empty

    forward = response_pairs[["stimulus_left", "stimulus_right", "mean_distance"]].rename(
        columns={"stimulus_left": "stimulus_row", "stimulus_right": "stimulus_column"}
    )
    reverse = response_pairs.loc[response_pairs["stimulus_left"] != response_pairs["stimulus_right"]].rename(
        columns={"stimulus_right": "stimulus_row", "stimulus_left": "stimulus_column"}
    )[["stimulus_row", "stimulus_column", "mean_distance"]]
    matrix = pd.concat([forward, reverse], ignore_index=True).pivot(
        index="stimulus_row",
        columns="stimulus_column",
        values="mean_distance",
    )
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    pd.testing.assert_frame_equal(matrix, matrix.T, check_dtype=False, check_names=False)


def test_split_half_reliability_is_reproducible(stage1_trial_metadata, stage1_trial_tensor):
    run_split_half_reliability = getattr(reliability_module, "run_split_half_reliability", None)
    assert run_split_half_reliability is not None, "run_split_half_reliability is not implemented yet"

    metadata = reliability_module.add_individual_id(stage1_trial_metadata)
    views = reliability_module.build_trial_views(metadata, stage1_trial_tensor)
    first_repeat, first_summary = run_split_half_reliability(views["response_window"], n_repeats=5, seed=13)
    second_repeat, second_summary = run_split_half_reliability(views["response_window"], n_repeats=5, seed=13)

    pd.testing.assert_frame_equal(first_repeat.reset_index(drop=True), second_repeat.reset_index(drop=True), check_like=True)
    pd.testing.assert_frame_equal(first_summary.reset_index(drop=True), second_summary.reset_index(drop=True), check_like=True)

import pandas as pd
import pytest

from bacteria_analysis import reliability as reliability_module

try:
    from bacteria_analysis import reliability_stats as reliability_stats_module
except ImportError:  # pragma: no cover - collection-time fallback until the module lands
    reliability_stats_module = None


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise AssertionError(f"missing expected column; tried {candidates}, got {list(frame.columns)}")


def test_permutation_null_preserves_label_counts_and_iteration_count(
    synthetic_trial_metadata,
    synthetic_trial_tensor,
):
    assert reliability_stats_module is not None, "bacteria_analysis.reliability_stats is not implemented yet"

    views = reliability_module.build_trial_views(synthetic_trial_metadata, synthetic_trial_tensor)
    trial_view = views["full_trajectory"]
    trial_metadata = reliability_module.add_individual_id(trial_view.metadata)

    null = reliability_stats_module.build_permutation_null(trial_view, n_iterations=11, random_state=7)

    iteration_col = _pick_column(null, ("iteration", "permutation_iteration", "perm_iteration"))
    label_col = _pick_column(null, ("stimulus", "permuted_stimulus", "shuffled_stimulus", "label"))
    trial_col = _pick_column(null, ("trial_id", "trial_id_left", "trial_id_a"))

    assert null[iteration_col].nunique() == 11
    assert null[trial_col].nunique() == len(trial_metadata)

    observed_counts = trial_metadata["stimulus"].value_counts().sort_index()
    for _, sample in null.groupby(iteration_col, sort=False):
        permuted_counts = sample[label_col].value_counts().sort_index()
        pd.testing.assert_series_equal(permuted_counts, observed_counts, check_names=False)


def test_grouped_bootstrap_resamples_individuals_without_splitting(
    synthetic_trial_metadata,
    synthetic_trial_tensor,
):
    assert reliability_stats_module is not None, "bacteria_analysis.reliability_stats is not implemented yet"

    views = reliability_module.build_trial_views(synthetic_trial_metadata, synthetic_trial_tensor)
    trial_view = views["full_trajectory"]
    trial_metadata = reliability_module.add_individual_id(trial_view.metadata)

    bootstrap = reliability_stats_module.build_grouped_bootstrap(trial_view, group_column="individual_id", n_iterations=9, random_state=3)

    iteration_col = _pick_column(bootstrap, ("iteration", "bootstrap_iteration", "resample_iteration"))
    score_col = _pick_column(bootstrap, ("score", "bootstrap_score", "statistic"))
    sampled_groups_col = None
    for candidate in ("sampled_individual_ids", "bootstrap_individual_ids", "sampled_groups"):
        if candidate in bootstrap.columns:
            sampled_groups_col = candidate
            break

    assert bootstrap[iteration_col].nunique() == 9
    assert score_col in bootstrap.columns

    expected_individual_ids = set(trial_metadata["individual_id"].unique())

    if sampled_groups_col is not None:
        for sampled_groups in bootstrap[sampled_groups_col]:
            assert set(sampled_groups).issubset(expected_individual_ids)
    else:
        assert "individual_id" in bootstrap.columns
        for _, sample in bootstrap.groupby(iteration_col, sort=False):
            assert set(sample["individual_id"]).issubset(expected_individual_ids)
            counts = sample["individual_id"].value_counts()
            assert (counts % 3 == 0).all()


def test_grouped_bootstrap_from_scores_uses_group_mean_estimand():
    assert reliability_stats_module is not None, "bacteria_analysis.reliability_stats is not implemented yet"

    trial_scores = pd.DataFrame(
        [
            {"view_name": "full_trajectory", "heldout_group": "g1", "score_status": "scored", "is_correct": 1},
            {"view_name": "full_trajectory", "heldout_group": "g1", "score_status": "scored", "is_correct": 1},
            {"view_name": "full_trajectory", "heldout_group": "g2", "score_status": "scored", "is_correct": 0},
            {"view_name": "full_trajectory", "heldout_group": "g2", "score_status": "scored", "is_correct": 0},
            {"view_name": "full_trajectory", "heldout_group": "g2", "score_status": "scored", "is_correct": 0},
        ]
    )

    iterations, summary = reliability_stats_module.build_grouped_bootstrap_from_scores(
        trial_scores,
        n_iterations=1,
        random_state=4,
    )

    sampled_groups = iterations.loc[0, "sampled_groups"]
    expected_group_means = {
        "g1": 1.0,
        "g2": 0.0,
    }
    expected_score = sum(expected_group_means[group_id] for group_id in sampled_groups) / len(sampled_groups)

    assert iterations.loc[0, "score_mean"] == pytest.approx(expected_score)
    assert summary.loc[0, "bootstrap_mean"] == pytest.approx(expected_score)

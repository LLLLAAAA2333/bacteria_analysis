"""Focused synthetic checks; no experimental data or notebooks are executed."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trial_representation import fit_representation
from trial_comparisons import (
    _bootstrap_fit, _bootstrap_indices, aggregate_original_support, cosine_matrix,
    fixed_support_chord, heldout_fold, run_bootstrap_chord, run_comparisons,
)


def inputs_from_trials(trials):
    """trials[N,R,K,C,40], all NaN allowed for absent whole trial curves."""
    trials = np.asarray(trials, float)
    count = np.isfinite(trials).all(axis=-1).sum(axis=1)
    mean = np.divide(np.nansum(trials, axis=1), count[..., None],
                     out=np.full(trials.shape[:1] + trials.shape[2:], np.nan), where=count[..., None] > 0)
    second = np.divide(np.nansum(trials ** 2, axis=1), count[..., None],
                       out=np.full(mean.shape, np.nan), where=count[..., None] > 0)
    return mean, count, second


def synthetic_data(n_animals=6, n_conditions=3, n_cells=4):
    rng = np.random.default_rng(73)
    h = np.repeat(np.array([0., 1., 2., 1., 0., -.5, -.5, 0.]), 5)
    amplitudes = rng.uniform(.3, 1.4, (n_conditions, n_cells))
    amplitudes[1, 0] *= -1
    trials = amplitudes[None, None, ..., None] * h
    trials = trials + rng.normal(0., .025, (n_animals, 3, n_conditions, n_cells, 40))
    raw, counts, second = inputs_from_trials(trials)
    return dict(raw=raw, trial_counts=counts, trial_second_moment=second,
                baseline_sd=np.full(raw.shape[:-1], .03),
                animals=[f"20260101|{i}" for i in range(n_animals)],
                conditions=[(f"A{i+1:03d}", "20260101") for i in range(n_conditions)],
                cells=[f"c{i}" for i in range(n_cells)])


def fit_data(data, threshold=1., min_animals=2):
    return fit_representation(data["raw"], data["baseline_sd"], [s for s, _ in data["conditions"]],
                              threshold=threshold, min_animals=min_animals,
                              trial_counts=data["trial_counts"],
                              trial_second_moment=data["trial_second_moment"])


class TrialVarianceTests(unittest.TestCase):
    def test_unequal_trial_counts_match_explicit_balanced_weights(self):
        trials = np.full((2, 4, 1, 1, 40), np.nan)
        trials[0, :2, 0, 0] = np.array([4., 6.])[:, None]
        trials[1, :, 0, 0] = np.array([3., 5., 7., 9.])[:, None]
        raw, counts, second = inputs_from_trials(trials)
        fit = fit_representation(raw, None, ["a"], trial_counts=counts, trial_second_moment=second)
        values = np.array([4., 6., 3., 5., 7., 9.])
        weights = np.array([.25, .25, .125, .125, .125, .125])
        mean = np.sum(values * weights)
        sw2 = np.sum(weights ** 2)
        variance = np.sum(weights * (values - mean) ** 2) / (1 - sw2)
        coherent = mean ** 2 - variance * sw2
        self.assertAlmostEqual(float(fit["means"][0, 0, 0]), mean)
        self.assertAlmostEqual(float(fit["sum_squared_trial_weights"][0, 0]), sw2)
        self.assertAlmostEqual(float(fit["effective_n"][0, 0]), 1 / sw2)
        self.assertAlmostEqual(float(fit["scatter_power"][0, 0]), variance)
        self.assertAlmostEqual(float(fit["coherent_power"][0, 0]), coherent)
        self.assertAlmostEqual(float(fit["snr"][0, 0]), np.sqrt(coherent / variance))
        self.assertEqual(fit["n_trials"][0, 0], 6)

    def test_within_animal_trial_noise_is_not_hidden_by_identical_means(self):
        trials = np.broadcast_to(np.array([-1.8, 2.2])[None, :, None, None, None], (3, 2, 1, 1, 40)).copy()
        raw, counts, second = inputs_from_trials(trials)
        fit = fit_representation(raw, None, ["a"], trial_counts=counts, trial_second_moment=second)
        self.assertTrue(np.all(raw == raw[0]))
        self.assertGreater(fit["scatter_power"][0, 0], 0.)
        self.assertEqual(fit["status"][0, 0], "below_snr")
        self.assertEqual(fit["coefficients"][0, 0], 0.)

    def test_units_and_animal_order_preserve_gate(self):
        data = synthetic_data()
        first = fit_data(data)
        scaled = {**data, "raw": data["raw"][::-1] * 10.,
                  "trial_counts": data["trial_counts"][::-1],
                  "trial_second_moment": data["trial_second_moment"][::-1] * 100.,
                  "baseline_sd": data["baseline_sd"][::-1] * 10.}
        second = fit_data(scaled)
        np.testing.assert_allclose(first["snr"], second["snr"], rtol=1e-10)
        np.testing.assert_array_equal(first["status"], second["status"])
        np.testing.assert_allclose(first["templates"], second["templates"], atol=1e-10)
        np.testing.assert_allclose(first["coefficients"] * 10, second["coefficients"], atol=1e-10)

    def test_trial_count_requirement_and_single_animal_training(self):
        raw = np.ones((1, 1, 1, 40))
        for trial_count, expected in ((2, "limited_trials"), (3, "retained")):
            fit = fit_representation(raw, None, ["a"], min_animals=1,
                                     trial_counts=np.full(raw.shape[:-1], trial_count),
                                     trial_second_moment=raw ** 2 + .01)
            self.assertEqual(fit["status"][0, 0], expected)
        full = fit_representation(raw, None, ["a"], min_animals=2,
                                  trial_counts=np.full(raw.shape[:-1], 3),
                                  trial_second_moment=raw ** 2 + .01)
        self.assertEqual(full["status"][0, 0], "limited_n")
        self.assertTrue(np.isnan(full["coefficients"][0, 0]))

    def test_zero_and_missing_remain_distinct(self):
        raw = np.zeros((3, 2, 1, 40))
        raw[:, 1] = np.nan
        counts = np.full(raw.shape[:-1], 3)
        counts[:, 1] = 0
        for threshold in (1., None):
            fit = fit_representation(raw, None, ["a", "b"], threshold=threshold,
                                     trial_counts=counts, trial_second_moment=raw ** 2)
            self.assertEqual(fit["coefficients"][0, 0], 0.)
            self.assertTrue(np.all(fit["reconstruction"][0, 0] == 0.))
            self.assertTrue(np.isnan(fit["coefficients"][1, 0]))
            similarity, _ = cosine_matrix(fit["coefficients"], min_shared=1)
            self.assertTrue(np.isnan(similarity).all())


class HeldAnimalTests(unittest.TestCase):
    def test_held_animal_and_nested_trial_stats_do_not_change_prediction(self):
        data = synthetic_data(n_animals=4)
        ids = [s for s, _ in data["conditions"]]
        first = heldout_fold(data["raw"], data["baseline_sd"], ids, 0,
                              trial_counts=data["trial_counts"], trial_second_moment=data["trial_second_moment"])
        raw = data["raw"].copy()
        counts = data["trial_counts"].copy()
        second = data["trial_second_moment"].copy()
        raw[0] += 100.
        counts[0] += 1000
        second[0] = raw[0] ** 2 + 500.
        other = heldout_fold(raw, data["baseline_sd"], ids, 0,
                             trial_counts=counts, trial_second_moment=second)
        for name in ("filtered_fit", "unfiltered_fit"):
            for key in ("templates", "coefficients", "snr", "reconstruction", "n_trials"):
                np.testing.assert_allclose(first[name][key], other[name][key], equal_nan=True)
        np.testing.assert_allclose(other["actual"] - first["actual"], 100.)

    def test_gate_failed_training_keeps_nonzero_held_target(self):
        raw = np.zeros((3, 2, 1, 40))
        raw[:, 1] = 1.
        raw[0, 0] = 10.
        counts = np.full(raw.shape[:-1], 3)
        second = raw ** 2 + 1.
        fold = heldout_fold(raw, None, ["a", "b"], 0,
                            trial_counts=counts, trial_second_moment=second)
        self.assertEqual(fold["filtered_fit"]["status"][0, 0], "below_snr")
        self.assertTrue(fold["score_mask"][0, 0])
        np.testing.assert_array_equal(fold["filtered_fit"]["reconstruction"][0, 0], np.zeros(8))
        np.testing.assert_array_equal(fold["actual"][0, 0], np.full(8, 10.))


class BootstrapSupportTests(unittest.TestCase):
    def test_whole_animal_draws_remain_date_stratified(self):
        animals = ["d1|a", "d1|b", "d2|c", "d2|d", "d2|e"]
        draw = _bootstrap_indices(animals, np.random.default_rng(7))
        self.assertEqual(len(draw), 5)
        self.assertTrue(np.all(draw[:2] < 2))
        self.assertTrue(np.all(draw[2:] >= 2))

    def test_lost_original_date_is_not_averaged_away(self):
        conditions = [("a", "d1"), ("a", "d2"), ("b", "d1")]
        original = np.array([[True, True], [True, False], [True, True]])
        values = np.array([[1., 2.], [np.nan, 999.], [3., 4.]])
        result, strains = aggregate_original_support(values, conditions, original)
        self.assertEqual(strains, ["a", "b"])
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertEqual(result[0, 1], 2.)

    def test_bootstrap_duplicates_cannot_create_original_support(self):
        data = synthetic_data(n_animals=3, n_conditions=2, n_cells=1)
        data["raw"][1:, 0] = np.nan
        data["trial_counts"][1:, 0] = 0
        data["trial_second_moment"][1:, 0] = np.nan
        data["baseline_sd"][1:, 0] = np.nan
        full = fit_data(data)
        support = np.isfinite(full["coefficients"])
        self.assertFalse(support[0, 0])
        boot = _bootstrap_fit(data, np.array([0, 0, 0]), support)
        self.assertEqual(boot["counts"][0, 0], 0)
        self.assertTrue(np.isnan(boot["coefficients"][0, 0]))
        self.assertTrue(np.isfinite(boot["coefficients"][1, 0]))

    def test_pair_mask_does_not_shrink_when_required_cell_is_lost(self):
        values = np.array([[1., 2., 3., 4., 5.], [2., 3., 4., 5., 6.]])
        support = np.ones(values.shape, bool)
        original = fixed_support_chord(values, support)
        self.assertTrue(np.isfinite(original[0, 1]))
        values[0, 4] = np.nan
        draw = fixed_support_chord(values, support)
        self.assertTrue(np.isnan(draw[0, 1]))
        values[0] = 0.
        self.assertTrue(np.isnan(fixed_support_chord(values, support)[0, 1]))

    def test_small_pipeline_and_bootstrap_write_compatible_outputs(self):
        data = synthetic_data()
        fit = fit_data(data)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chem = root / "reports/population_first_20260930/tables"
            chem.mkdir(parents=True)
            pd.DataFrame([[1., 2.], [2., 1.], [3., 4.]], index=["A001", "A002", "A003"]).to_csv(
                chem / "aligned_chemical_log2fc_paired.csv")
            facts = run_comparisons(data, fit, root / "output", root / "reports", repeats=2, progress=None)
            self.assertEqual(facts["loao"]["n_scored"], 72)
            bootstrap = run_bootstrap_chord(data, fit, root / "output", draws=5, progress=None)
            self.assertEqual(bootstrap["n_eligible_pairs"], 3)
            tables = root / "output/tables"
            mask = pd.read_csv(tables / "hmds_neural_pair_mask.csv", index_col=0).to_numpy(bool)
            variance = pd.read_csv(tables / "hmds_neural_variance.csv", index_col=0).to_numpy(float)
            self.assertFalse(np.diag(mask).any())
            self.assertTrue(np.all(variance[mask] >= 0.))
            np.testing.assert_array_equal(mask, mask.T)


if __name__ == "__main__":
    unittest.main()

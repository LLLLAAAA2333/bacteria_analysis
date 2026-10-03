"""Scientific invariants for individual-SNR comparisons; synthetic data only."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from individual_representation import fit_representation
from individual_comparisons import (
    _bootstrap_fit, _bootstrap_indices, aggregate_original_support, cosine_matrix,
    fixed_support_chord, heldout_fold, run_bootstrap_chord, run_comparisons,
)


def synthetic_data(n_animals=6, n_conditions=3, n_cells=4):
    rng = np.random.default_rng(73)
    h = np.repeat(np.array([0., 1., 2., 1., 0., -.5, -.5, 0.]), 5)
    amplitudes = rng.uniform(.3, 1.4, (n_conditions, n_cells))
    amplitudes[1, 0] *= -1
    raw = amplitudes[None, ..., None] * h + rng.normal(0., .025, (n_animals, n_conditions, n_cells, 40))
    return dict(raw=raw, baseline_sd=np.full(raw.shape[:-1], .03),
                animals=[f"20260101|{i}" for i in range(n_animals)],
                conditions=[(f"A{i+1:03d}", "20260101") for i in range(n_conditions)],
                cells=[f"c{i}" for i in range(n_cells)])


def fit_data(data, threshold=.5, min_animals=2):
    return fit_representation(data["raw"], data["baseline_sd"], [s for s, _ in data["conditions"]],
                              threshold=threshold, min_animals=min_animals)


class IndividualVarianceTests(unittest.TestCase):
    def test_individual_scatter_formula_and_lower_cutoff(self):
        raw = np.array([1-np.sqrt(.75), 1-np.sqrt(.75), 1+np.sqrt(.75), 1+np.sqrt(.75)])
        raw = np.broadcast_to(raw[:, None, None, None], (4, 1, 1, 40)).copy()
        lower = fit_representation(raw, None, ["a"], threshold=.5, min_animals=2)
        higher = fit_representation(raw, None, ["a"], threshold=1., min_animals=2)
        self.assertAlmostEqual(lower["scatter_power"][0, 0], 1.)
        self.assertAlmostEqual(lower["coherent_power"][0, 0], .75)
        self.assertAlmostEqual(lower["snr"][0, 0], np.sqrt(.75))
        self.assertEqual(lower["status"][0, 0], "retained")
        self.assertEqual(higher["status"][0, 0], "below_snr")
        self.assertNotEqual(lower["coefficients"][0, 0], 0.)
        self.assertEqual(higher["coefficients"][0, 0], 0.)

    def test_units_and_animal_order_preserve_gate(self):
        data = synthetic_data()
        first = fit_data(data)
        scaled = {**data, "raw": data["raw"][::-1] * 10., "baseline_sd": data["baseline_sd"][::-1] * 10.}
        second = fit_data(scaled)
        np.testing.assert_allclose(first["snr"], second["snr"], rtol=1e-10)
        np.testing.assert_array_equal(first["status"], second["status"])
        np.testing.assert_allclose(first["templates"], second["templates"], atol=1e-10)
        np.testing.assert_allclose(first["coefficients"] * 10, second["coefficients"], atol=1e-10)

    def test_zero_missing_and_one_animal_are_distinct(self):
        raw = np.zeros((3, 3, 1, 40))
        raw[:, 1] = np.nan
        raw[1:, 2] = np.nan
        for threshold in (.5, None):
            fit = fit_representation(raw, None, list("abc"), threshold=threshold, min_animals=2)
            self.assertEqual(fit["coefficients"][0, 0], 0.)
            self.assertTrue(np.all(fit["reconstruction"][0, 0] == 0.))
            self.assertEqual(fit["status"][1, 0], "missing")
            self.assertEqual(fit["status"][2, 0], "limited_n")
            self.assertTrue(np.isnan(fit["coefficients"][1:, 0]).all())
            similarity, _ = cosine_matrix(fit["coefficients"], min_shared=1)
            self.assertTrue(np.isnan(similarity).all())


class HeldAnimalTests(unittest.TestCase):
    def test_held_animal_cannot_change_training_gate_or_template(self):
        data = synthetic_data(n_animals=4)
        ids = [s for s, _ in data["conditions"]]
        first = heldout_fold(data["raw"], data["baseline_sd"], ids, 0, primary_threshold=.5)
        raw, baseline = data["raw"].copy(), data["baseline_sd"].copy()
        raw[0] += 100.
        baseline[0] *= 1000
        other = heldout_fold(raw, baseline, ids, 0, primary_threshold=.5)
        for name in ("filtered_fit", "unfiltered_fit"):
            for key in ("templates", "coefficients", "snr", "reconstruction"):
                np.testing.assert_allclose(first[name][key], other[name][key], equal_nan=True)
        np.testing.assert_allclose(other["actual"] - first["actual"], 100.)

    def test_threshold_reaches_heldout_and_bootstrap_fits(self):
        raw = np.ones((4, 2, 1, 40))
        raw[:, 0, 0] = np.array([10., 0., 1., 2.])[:, None]
        data = dict(raw=raw, baseline_sd=None, conditions=[("a", "d"), ("b", "d")])
        lower = heldout_fold(raw, None, ["a", "b"], 0, primary_threshold=.5)
        higher = heldout_fold(raw, None, ["a", "b"], 0, primary_threshold=1.)
        self.assertEqual(lower["filtered_fit"]["status"][0, 0], "retained")
        self.assertEqual(higher["filtered_fit"]["status"][0, 0], "below_snr")
        self.assertTrue(higher["score_mask"][0, 0])
        np.testing.assert_array_equal(higher["filtered_fit"]["reconstruction"][0, 0], np.zeros(8))
        np.testing.assert_array_equal(higher["actual"][0, 0], np.full(8, 10.))
        support = np.ones((2, 1), bool)
        low_boot = _bootstrap_fit(data, np.array([1, 2, 3]), support, primary_threshold=.5)
        high_boot = _bootstrap_fit(data, np.array([1, 2, 3]), support, primary_threshold=1.)
        self.assertEqual(low_boot["status"][0, 0], "retained")
        self.assertEqual(high_boot["status"][0, 0], "below_snr")

    def test_only_one_training_animal_stays_unscored(self):
        raw = np.ones((2, 1, 1, 40))
        fold = heldout_fold(raw, None, ["a"], 0)
        self.assertEqual(fold["filtered_fit"]["status"][0, 0], "limited_n")
        self.assertFalse(fold["score_mask"][0, 0])
        self.assertTrue(np.isnan(fold["filtered_fit"]["reconstruction"][0, 0]).all())


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
        self.assertTrue(np.isfinite(fixed_support_chord(values, support)[0, 1]))
        values[0, 4] = np.nan
        self.assertTrue(np.isnan(fixed_support_chord(values, support)[0, 1]))
        values[0] = 0.
        self.assertTrue(np.isnan(fixed_support_chord(values, support)[0, 1]))

    def test_pipeline_and_bootstrap_propagate_nondefault_primary(self):
        data = synthetic_data()
        fit = fit_data(data, threshold=.75)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chem = root / "reports/population_first_20260930/tables"
            chem.mkdir(parents=True)
            pd.DataFrame([[1., 2.], [2., 1.], [3., 4.]], index=["A001", "A002", "A003"]).to_csv(
                chem / "aligned_chemical_log2fc_paired.csv")
            with patch("individual_comparisons.fit_representation", wraps=fit_representation) as tracked:
                facts = run_comparisons(data, fit, root / "output", root / "reports",
                                        repeats=2, progress=None, primary_threshold=.75)
                bootstrap = run_bootstrap_chord(data, fit, root / "output", draws=5,
                                                progress=None, primary_threshold=.75)
            self.assertTrue(tracked.call_args_list)
            self.assertTrue(all(call.kwargs["threshold"] in (.75, None) for call in tracked.call_args_list))
            self.assertTrue(all(call.kwargs["min_animals"] == 2 for call in tracked.call_args_list))
            self.assertEqual(facts["loao"]["n_scored"], 72)
            self.assertEqual(facts["split"]["primary_threshold"], .75)
            self.assertEqual(bootstrap["primary_threshold"], .75)
            self.assertEqual(bootstrap["n_eligible_pairs"], 3)
            tables = root / "output/tables"
            mask = pd.read_csv(tables / "hmds_neural_pair_mask.csv", index_col=0).to_numpy(bool)
            variance = pd.read_csv(tables / "hmds_neural_variance.csv", index_col=0).to_numpy(float)
            self.assertFalse(np.diag(mask).any())
            self.assertTrue(np.all(variance[mask] >= 0.))
            np.testing.assert_array_equal(mask, mask.T)
            with self.assertRaises(ValueError):
                run_bootstrap_chord(data, fit, root / "mismatch", draws=2, progress=None,
                                    primary_threshold=.5)


if __name__ == "__main__":
    unittest.main()

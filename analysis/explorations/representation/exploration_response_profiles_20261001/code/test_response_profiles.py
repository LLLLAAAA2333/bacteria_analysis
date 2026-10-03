"""Focused scientific invariants for the response-profile exploration.

Synthetic curves only: these tests do not load or process experimental data.
Run with the repository's Python interpreter and unittest.
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from response_comparisons import cosine_matrix, heldout_fold, split_cosine
from response_representation import aggregate_strains, bin_curves, fit_representation


def _raw_from_bins(binned):
    """Convert [animal, condition, cell, 8] to forty 1-s samples."""
    return np.repeat(np.asarray(binned, dtype=float), 5, axis=-1)


class ResponseRepresentationTests(unittest.TestCase):
    def test_filter_preserves_sparse_signal_and_distinguishes_missing(self):
        h = np.array([0., 1., 2., 1., 0., -.5, -.5, 0.])
        bins = np.zeros((4, 3, 2, 8))
        bins[:, 0, 0] = np.array([.98, 1., 1.02, 1.])[:, None] * h
        bins[:, 1, 0] = np.array([-1., 1., -1., 1.])[:, None] * h
        bins[:, 2, 0] = np.nan
        # A rarely responsive cell must not be discarded as a whole merely
        # because the other conditions contain only noise.
        bins[:, :2, 1] = np.array([-1., 1., -1., 1.])[:, None, None] * h
        bins[:, 2, 1] = 2. * h
        fit = fit_representation(_raw_from_bins(bins), None, list("abc"))
        self.assertEqual(fit["status"][0, 0], "retained")
        self.assertEqual(fit["status"][2, 1], "retained")
        self.assertEqual(fit["status"][1, 0], "below_snr")
        self.assertEqual(fit["coefficients"][1, 0], 0.)
        np.testing.assert_array_equal(fit["reconstruction"][1, 0], np.zeros(8))
        self.assertEqual(fit["status"][2, 0], "missing")
        self.assertTrue(np.isnan(fit["coefficients"][2, 0]))
        self.assertTrue(np.isnan(fit["reconstruction"][2, 0]).all())

    def test_no_signal_cell_keeps_measured_zero_without_inventing_shape(self):
        raw = np.zeros((4, 2, 1, 40))
        for threshold in (1., None):
            with self.subTest(threshold=threshold):
                fit = fit_representation(raw, None, ["a", "b"], threshold=threshold)
                np.testing.assert_array_equal(fit["coefficients"], np.zeros((2, 1)))
                np.testing.assert_array_equal(fit["reconstruction"], np.zeros((2, 1, 8)))
                self.assertFalse(fit["template_identified"][0])

    def test_minimum_animal_count_is_not_replaced_with_zero(self):
        raw = np.ones((3, 2, 1, 40))
        raw[2, 1] = np.nan
        full = fit_representation(raw, None, ["a", "b"], min_animals=3)
        train = fit_representation(raw, None, ["a", "b"], min_animals=2)
        self.assertEqual(full["counts"][1, 0], 2)
        self.assertEqual(full["status"][1, 0], "limited_n")
        self.assertTrue(np.isnan(full["coefficients"][1, 0]))
        self.assertTrue(np.isfinite(train["coefficients"][1, 0]))

    def test_rank_one_signal_reconstructs_exactly_with_unit_rms_shape(self):
        h = np.array([0., .5, 2., 1., -.5, -.2, -.1, 0.])
        amplitude = np.array([-2., 1., 3.])
        means = amplitude[:, None, None] * h[None, None, :]
        raw = _raw_from_bins(np.broadcast_to(means, (4, 3, 1, 8)))
        fit = fit_representation(raw, None, list("abc"))
        np.testing.assert_allclose(fit["reconstruction"], means, atol=1e-10)
        np.testing.assert_allclose(np.mean(fit["templates"] ** 2, axis=1), 1.)
        templates = fit["templates"]
        for row in templates:
            self.assertGreater(row[np.argmax(np.abs(row))], 0.)
        expected = np.sum(fit["mean_bins"] * templates[None], axis=-1) / np.sum(
            templates ** 2, axis=-1)[None]
        np.testing.assert_allclose(fit["raw_coefficients"], expected, atol=1e-10)
        np.testing.assert_allclose(fit["coefficients"], expected, atol=1e-10)

    def test_primary_snr_is_invariant_to_response_units(self):
        rng = np.random.default_rng(43)
        raw = rng.normal(0., .08, (5, 3, 2, 40))
        raw[:, 0] += .5
        raw[:, 2, 1] -= .7
        baseline = np.full(raw.shape[:-1], .04)
        before = fit_representation(raw, baseline, list("abc"))
        after = fit_representation(10. * raw, 10. * baseline, list("abc"))
        np.testing.assert_array_equal(before["status"], after["status"])
        np.testing.assert_allclose(before["snr"], after["snr"], atol=1e-12)
        np.testing.assert_allclose(before["templates"], after["templates"], atol=1e-10)
        np.testing.assert_allclose(10. * before["coefficients"], after["coefficients"],
                                   atol=1e-10, equal_nan=True)

    def test_snr_calibration_and_single_animal_outlier(self):
        # Constant curves of heights 1, 2, 3 have mean-power 4 and sample
        # variance 1, hence corrected signal 4 - 1/3. A response confined to
        # one animal instead has zero corrected signal in this estimator.
        raw = np.zeros((3, 2, 1, 40))
        raw[:, 0, 0] = np.array([1., 2., 3.])[:, None]
        raw[:, 1, 0] = np.array([9., 0., 0.])[:, None]
        fit = fit_representation(raw, None, ["a", "b"])
        self.assertAlmostEqual(fit["snr"][0, 0], np.sqrt(11. / 3.))
        self.assertEqual(fit["snr"][1, 0], 0.)
        self.assertEqual(fit["status"][1, 0], "below_snr")

    def test_strain_aggregation_uses_equal_dates_without_imputing_missing(self):
        values = np.array([[1., np.nan, np.nan], [3., 5., np.nan], [7., 9., 0.]])
        result, strains = aggregate_strains(values, [("a", "d1"), ("a", "d2"),
                                                     ("b", "d3")])
        self.assertEqual(list(strains), ["a", "b"])
        np.testing.assert_allclose(result, [[2., 5., np.nan], [7., 9., 0.]],
                                   equal_nan=True)

    def test_partial_curve_is_not_silently_treated_as_complete(self):
        raw = np.ones((2, 40))
        raw[1, 3] = np.nan
        with self.assertRaises(ValueError):
            bin_curves(raw)
        raw[1] = np.nan
        binned = bin_curves(raw)
        np.testing.assert_array_equal(binned[0], np.ones(8))
        self.assertTrue(np.isnan(binned[1]).all())


class HeldAnimalTests(unittest.TestCase):
    def test_held_animal_cannot_change_its_training_gate_or_template(self):
        rng = np.random.default_rng(27)
        raw = rng.normal(0., .1, (4, 3, 2, 40))
        raw[:, 0, 0] += 1.
        raw[:, 1, 1] += .8
        baseline = np.full(raw.shape[:-1], .1)
        before = heldout_fold(raw, baseline, list("abc"), held_index=0, threshold=1.)
        changed = raw.copy()
        changed[0] += 100.
        changed_baseline = baseline.copy()
        changed_baseline[0] *= 1000.
        after = heldout_fold(changed, changed_baseline, list("abc"),
                             held_index=0, threshold=1.)
        for name in ("filtered_fit", "unfiltered_fit"):
            for key in ("templates", "coefficients", "snr", "reconstruction"):
                np.testing.assert_allclose(before[name][key], after[name][key],
                                           atol=1e-10, equal_nan=True)
            np.testing.assert_array_equal(before[name]["status"], after[name]["status"])
        np.testing.assert_array_equal(before["score_mask"], after["score_mask"])
        np.testing.assert_allclose(after["actual"] - before["actual"], 100.)

    def test_failed_training_gate_is_scored_against_raw_held_response(self):
        h = np.array([0., 1., 2., 1., 0., -.5, -.5, 0.])
        bins = np.empty((4, 2, 1, 8))
        bins[:, 0, 0] = np.array([10., -1., 0., 1.])[:, None] * h
        bins[:, 1, 0] = 2. * h
        fold = heldout_fold(_raw_from_bins(bins), None, ["a", "b"],
                            held_index=0, threshold=1.)
        self.assertEqual(fold["filtered_fit"]["status"][0, 0], "below_snr")
        self.assertTrue(fold["score_mask"][0, 0])
        np.testing.assert_array_equal(fold["filtered_fit"]["reconstruction"][0, 0],
                                      np.zeros(8))
        np.testing.assert_array_equal(fold["actual"][0, 0], 10. * h)

    def test_silent_training_cell_does_not_hide_a_held_response(self):
        raw = np.zeros((4, 1, 1, 40))
        raw[0] = 2.
        fold = heldout_fold(raw, None, ["a"], held_index=0, threshold=1.)
        self.assertTrue(fold["score_mask"][0, 0])
        np.testing.assert_array_equal(fold["filtered_fit"]["reconstruction"][0, 0],
                                      np.zeros(8))
        np.testing.assert_array_equal(fold["actual"][0, 0], np.full(8, 2.))


class ResponseComparisonTests(unittest.TestCase):
    def test_all_zero_observation_has_undefined_cosine(self):
        coefficients = np.array([[0., 0., 0., 0.], [1., 2., 3., 4.]])
        similarity, shared = cosine_matrix(coefficients, min_shared=4)
        self.assertEqual(shared[0, 1], 4)
        self.assertTrue(np.isnan(similarity[0, 0]))
        self.assertTrue(np.isnan(similarity[0, 1]))
        self.assertAlmostEqual(similarity[1, 1], 1.)

    def test_distance_uses_only_shared_finite_cells(self):
        coefficients = np.array([[1., 2., 3., 4., np.nan],
                                 [1., 2., 3., 4., 1000.]])
        similarity, shared = cosine_matrix(coefficients, min_shared=4)
        self.assertEqual(shared[0, 1], 4)
        self.assertAlmostEqual(similarity[0, 1], 1.)
        excluded, _ = cosine_matrix(coefficients, min_shared=5)
        self.assertTrue(np.isnan(excluded[0, 1]))

    def test_fixed_unit_rms_templates_preserve_cosine_geometry(self):
        # Signed coefficients and distinct cell templates ensure this checks
        # actual geometry rather than only positive proportional vectors.
        templates = np.array([[1., 0., 1., 0., -1., 0., -1., 0.],
                              [0., 1., 2., 1., 0., -1., -2., -1.],
                              [1., 1., 1., 1., 1., 1., 1., 1.],
                              [-2., -1., 0., 1., 2., 1., 0., -1.]])
        templates /= np.sqrt(np.mean(templates ** 2, axis=1))[:, None]
        coefficients = np.array([[1., -2., .5, 0.], [.5, 1., 0., -3.],
                                 [np.nan, -1., 2., .5]])
        reconstructed = coefficients[:, :, None] * templates[None]
        coeff_cos, coeff_n = cosine_matrix(coefficients, min_shared=3)
        curve_cos, curve_n = cosine_matrix(reconstructed, min_shared=3)
        np.testing.assert_array_equal(coeff_n, curve_n)
        np.testing.assert_allclose(coeff_cos, curve_cos, atol=1e-12,
                                   equal_nan=True)

    def test_split_comparison_requires_four_way_cell_support(self):
        first = np.array([[1., 2., 3., 4., 5.], [2., 1., 4., 3., 5.]])
        second = first.copy()
        second[1, 4] = np.nan
        a = first[:, :, None] * np.ones((1, 1, 8))
        b = second[:, :, None] * np.ones((1, 1, 8))
        similarity, shared = split_cosine(a, b, min_shared=4)
        expected = np.dot(first[0, :4], first[1, :4]) / (
            np.linalg.norm(first[0, :4]) * np.linalg.norm(first[1, :4]))
        self.assertEqual(shared[0, 1], 4)
        self.assertAlmostEqual(similarity[0, 1], expected)
        np.testing.assert_allclose(similarity, similarity.T, equal_nan=True)

    def test_split_support_mask_and_zero_cosine_are_respected(self):
        a = np.ones((2, 5, 8))
        b = a.copy()
        support = np.ones((2, 5), dtype=bool)
        support[0, 4] = False
        _, shared = split_cosine(a, b, min_shared=4, support=support)
        self.assertEqual(shared[0, 1], 4)
        b[0] = 0.
        similarity, _ = split_cosine(a, b, min_shared=4, support=support)
        self.assertTrue(np.isnan(similarity[0, 0]))
        self.assertTrue(np.isnan(similarity[0, 1]))


if __name__ == "__main__":
    unittest.main()

"""Scientific invariants for fixed-template paired animal amplitude differences."""
import unittest

import numpy as np

from bacteria_analysis.neighborhood_amplitudes import (
    cross_animal,
    paired_statistics,
    project_complete_curves,
)


class NeighborhoodAmplitudeTests(unittest.TestCase):
    def test_known_signed_amplitudes_keep_original_units(self):
        h = np.array([1., 2., 3., 2., -1., -2., -.5, .5])
        h /= np.sqrt(np.mean(h ** 2))
        amplitudes = np.array([-1.25, 0., .375, 2.5])
        curves = amplitudes[:, None] * h
        np.testing.assert_allclose(
            project_complete_curves(curves, h), amplitudes, atol=1e-14)
        # A nonzero constant trace is not time-centered or fitted with an intercept.
        np.testing.assert_allclose(
            project_complete_curves(np.full((2, 8), 3.5), np.ones(8)),
            [3.5, 3.5])

    def test_all_missing_stays_missing_and_zero_stays_zero(self):
        curves = np.vstack([np.full(8, np.nan), np.zeros(8), np.ones(8)])
        result = project_complete_curves(curves, np.ones(8))
        self.assertTrue(np.isnan(result[0]))
        np.testing.assert_array_equal(result[1:], [0., 1.])

    def test_partial_bins_and_invalid_templates_are_rejected(self):
        for observed_bins in [1, 7]:
            with self.subTest(observed_bins=observed_bins):
                curve = np.full((1, 8), np.nan)
                curve[0, :observed_bins] = 0.
                with self.assertRaisesRegex(ValueError, "Partial-bin"):
                    project_complete_curves(curve, np.ones(8))
        with self.assertRaisesRegex(ValueError, "Infinite"):
            project_complete_curves(np.full((1, 8), np.inf), np.ones(8))
        for h in [np.ones(8) * 2, np.full(8, np.nan), np.ones(7)]:
            with self.subTest(template=h):
                with self.assertRaises(ValueError):
                    project_complete_curves(np.zeros((1, 8)), h)

    def test_paired_sem_and_negative_energy_keep_animal_signs(self):
        h = np.ones(8)
        b_amplitudes = np.array([2., 3., -1.])
        differences = np.array([1., -1., 1.])
        a = (b_amplitudes + differences)[:, None] * h
        b = b_amplitudes[:, None] * h
        result, aa, ab, delta, _, residual, deletions = paired_statistics(a, b, h)
        np.testing.assert_allclose(aa, [3., 2., 0.])
        np.testing.assert_allclose(ab, b_amplitudes)
        np.testing.assert_allclose(delta, differences)
        self.assertEqual(result["n_animals"], 3)
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["mean_delta"], 1 / 3)
        self.assertAlmostEqual(result["sem_delta"], 2 / 3)
        self.assertAlmostEqual(result["energy"], -1 / 3)
        self.assertEqual(result["n_delta_positive"], 2)
        self.assertEqual(result["n_delta_negative"], 1)
        np.testing.assert_array_equal(residual, np.zeros((3, 8)))
        np.testing.assert_allclose([row["energy"] for row in deletions], [-1, 1, -1])
        self.assertAlmostEqual(result["loo_energy_min"], -1.)
        self.assertAlmostEqual(result["loo_energy_max"], 1.)
        self.assertAlmostEqual(result["energy_identity_error"], 0.)

    def test_template_and_orthogonal_residual_energy_decompose(self):
        h = np.ones(8)
        orthogonal = np.tile([1., -1.], 4)
        delta = np.array([1., 2., 3.])
        residual_amplitude = np.array([2., 1., -1.])
        b = np.array([.5, -.25, 1.])[:, None] * h
        b += np.array([.25, .4, -.5])[:, None] * orthogonal
        a = b + delta[:, None] * h + residual_amplitude[:, None] * orthogonal
        result, _, _, actual_delta, curves, residual, _ = paired_statistics(a, b, h)
        np.testing.assert_allclose(actual_delta, delta)
        np.testing.assert_allclose(residual, residual_amplitude[:, None] * orthogonal)
        np.testing.assert_allclose(residual @ h, 0., atol=1e-14)
        self.assertAlmostEqual(result["energy"], 11 / 3)
        self.assertAlmostEqual(result["residual_energy"], -1 / 3)
        self.assertAlmostEqual(result["curve_energy"], 10 / 3)
        self.assertAlmostEqual(result["mean_delta"], 2.)
        self.assertAlmostEqual(result["mean_residual_rms"], 2 / 3)
        self.assertAlmostEqual(result["mean_curve_rms"] ** 2, 40 / 9)
        self.assertAlmostEqual(result["model_residual_fraction"], .1)
        self.assertAlmostEqual(result["decomposition_error"], 0.)
        self.assertAlmostEqual(result["energy_decomposition_error"], 0.)
        # Explicitly average the three unordered animal products independently.
        self.assertAlmostEqual(cross_animal(curves),
                               np.mean([np.mean(curves[i] * curves[j])
                                        for i, j in [(0, 1), (0, 2), (1, 2)]]))

    def test_pair_orientation_changes_direction_but_not_size_or_energy(self):
        h = np.ones(8)
        q = np.tile([1., -1.], 4)
        b = np.array([1., -.3, 2.])[:, None] * h
        a = b + np.array([.4, -.2, .7])[:, None] * h
        a += np.array([-.3, .2, .6])[:, None] * q
        forward = paired_statistics(a, b, h)
        reverse = paired_statistics(b, a, h)
        sf, sr = forward[0], reverse[0]
        self.assertAlmostEqual(sf["mean_delta"], -sr["mean_delta"])
        for name in ["abs_mean_delta", "sem_delta", "energy", "curve_energy",
                     "residual_energy", "mean_curve_rms", "mean_residual_rms",
                     "model_residual_fraction", "loo_energy_min", "loo_energy_max"]:
            with self.subTest(statistic=name):
                self.assertAlmostEqual(sf[name], sr[name])
        np.testing.assert_allclose(forward[1], reverse[2])
        np.testing.assert_allclose(forward[2], reverse[1])
        for index in [3, 4, 5]:
            np.testing.assert_allclose(forward[index], -reverse[index])
        self.assertAlmostEqual(sf["loo_mean_min"], -sr["loo_mean_max"])

    def test_low_coverage_retains_description_but_not_main_energy(self):
        h = np.ones(8)
        result, _, _, delta, _, _, deletions = paired_statistics(
            np.array([1., 3.])[:, None] * h, np.zeros((2, 8)), h)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["n_animals"], 2)
        self.assertAlmostEqual(result["mean_delta"], 2.)
        self.assertAlmostEqual(result["sem_delta"], 1.)
        self.assertTrue(np.isnan(result["energy"]))
        self.assertEqual(deletions, [])
        np.testing.assert_array_equal(delta, [1., 3.])
        # Two animals still define a distinct-animal product during deletion checks.
        self.assertAlmostEqual(cross_animal(delta), 3.)
        self.assertTrue(np.isnan(cross_animal(np.array([1.]))))

    def test_paired_inputs_cannot_silently_drop_or_impute_missing_animals(self):
        with self.assertRaisesRegex(ValueError, "identical finite"):
            paired_statistics(np.ones((3, 8)), np.ones((2, 8)), np.ones(8))
        a = np.ones((3, 8))
        a[1] = np.nan
        with self.assertRaisesRegex(ValueError, "identical finite"):
            paired_statistics(a, np.ones((3, 8)), np.ones(8))
        with self.assertRaisesRegex(ValueError, "complete finite"):
            cross_animal(np.array([1., np.nan, 3.]))


if __name__ == "__main__":
    unittest.main()

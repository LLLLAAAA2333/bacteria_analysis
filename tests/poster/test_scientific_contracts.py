"""Small-data contracts that protect the poster's scientific definitions.

Run from repository root:
    pixi run test

No report data, raw measurements, external processes, or figure generation needed.
"""
import unittest

import numpy as np
import pandas as pd

from poster_analysis.atlas import aggregate_conditions
from poster_analysis.repeatability import extract_distributions, histogram_summary
from poster_analysis.global_comparison import distance_matrices, paired_distances
from poster_analysis.chemical_axes import fit_axes, transform_axes


class ResponseContracts(unittest.TestCase):
    def test_condition_mean_preserves_observed_zero_and_missing_support(self):
        conditions = np.array([["A", "block1"], ["A", "block2"], ["B", "block1"]])
        values = np.array([[0., np.nan], [2., 8.], [9., 0.]])
        result = aggregate_conditions(values, conditions, ["B", "A", "unobserved"])
        # Equal condition weights: A's observed zero contributes to (0 + 2) / 2.
        # Missing support is not an additional zero and does not dilute the 8.
        np.testing.assert_allclose(result[:2], [[9., 0.], [1., 8.]])
        self.assertTrue(np.isnan(result[2]).all())
        self.assertTrue(np.isnan(values[0, 1]))  # input is not imputed in place

    def test_split_extraction_counts_each_unordered_pair_once(self):
        matrix = np.array([[.8, .2, np.nan], [.2, .9, -.4], [np.nan, -.4, np.nan]])
        result = extract_distributions(matrix)
        np.testing.assert_array_equal(result["same"], [.8, .9])
        np.testing.assert_array_equal(result["different"], [.2, -.4])
        self.assertEqual(result["undefined_same"], 1)
        self.assertEqual(result["undefined_different"], 1)
        asymmetric = matrix.copy()
        asymmetric[1, 0] = .3
        with self.assertRaises(ValueError):
            extract_distributions(asymmetric)

    def test_density_normalization_keeps_finite_tails(self):
        values = np.array([-1., -.25, 0., 1., np.nan])
        edges = np.array([-1., -.5, 0., .5, 1.])
        result = histogram_summary(values, edges)
        self.assertEqual(result["n"], 4)
        np.testing.assert_array_equal(result["counts"], [1, 1, 1, 1])
        self.assertAlmostEqual(float(np.sum(result["density"] * np.diff(edges))), 1.)
        with self.assertRaises(ValueError):
            histogram_summary(values, np.array([-.5, 0., .5]))


class GlobalDistanceContracts(unittest.TestCase):
    def setUp(self):
        self.chemical = pd.DataFrame({"m1": [0., 3., 6.], "m2": [0., 4., 8.]},
                                     index=["A", "B", "C"])
        self.neural = pd.DataFrame({"n1": [1., 0., -1.], "n2": [0., 2., 0.]},
                                   index=["A", "B", "C"])

    def test_known_geometry_and_identity_alignment(self):
        chemical, neural = distance_matrices(self.chemical.loc[["C", "A", "B"]], self.neural)
        step = 5. / np.sqrt(2.)
        np.testing.assert_allclose(chemical, [[0, step, 2*step], [step, 0, step], [2*step, step, 0]])
        # Cosine normalizes the B vector despite its length being two.
        np.testing.assert_allclose(neural, [[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        self.assertEqual(chemical.index.tolist(), ["A", "B", "C"])
        pairs = paired_distances(chemical, neural.loc[["B", "C", "A"], ["B", "C", "A"]])
        self.assertEqual(list(zip(pairs.strain_a, pairs.strain_b)), [("A", "B"), ("A", "C"), ("B", "C")])
        np.testing.assert_allclose(pairs.neural_distance, [1, 2, 1])

    def test_distances_invariant_to_independent_row_order(self):
        reference = distance_matrices(self.chemical, self.neural)
        reordered = distance_matrices(self.chemical.loc[["B", "A", "C"]],
                                      self.neural.loc[["C", "B", "A"]])
        for expected, actual in zip(reference, reordered):
            pd.testing.assert_frame_equal(actual.loc[expected.index, expected.columns], expected)

    def test_invalid_geometries_are_rejected_without_imputation(self):
        zero = self.neural.copy()
        zero.loc["A"] = 0.
        bad_chemical = self.chemical.copy()
        bad_chemical.loc["A", "m1"] = np.nan
        bad_neural = self.neural.copy()
        bad_neural.loc["B", "n2"] = np.inf
        for chemical, neural in [(self.chemical, zero), (bad_chemical, self.neural),
                                  (self.chemical, bad_neural), (self.chemical.iloc[:2], self.neural)]:
            with self.subTest(chemical_shape=chemical.shape, neural_values=neural.values.tolist()):
                with self.assertRaises(ValueError):
                    distance_matrices(chemical, neural)


class ChemicalAxisContracts(unittest.TestCase):
    def test_transform_uses_only_training_statistics_not_heldout_companions(self):
        x = np.array([-2., -1., 1., 2.])
        train = pd.DataFrame({"m1": 10+x, "m2": 20+2*x, "m3": 30+3*x, "constant": 55.},
                             index=["T1", "T2", "T3", "T4"])
        metadata = pd.DataFrame({"family": ["f1", "f2", "f3", "f4"]}, index=train.columns)
        fitted = fit_axes(train, metadata)
        self.assertEqual(fitted["train_ids"], train.index.tolist())
        self.assertEqual(fitted["excluded_features"], ["constant"])
        self.assertEqual(len(fitted["module_members"]), 1)
        means_before, scales_before = fitted["means"].copy(), fitted["scales"].copy()
        heldout = pd.DataFrame({"m1": [14.], "m2": [28.], "m3": [42.], "constant": [55.]}, index=["H"])
        alone = transform_axes(heldout, fitted)
        companions = pd.DataFrame({"m1": [1e6, -1e6], "m2": [-1e6, 1e6],
                                   "m3": [1e6, 1e6], "constant": [55., 55.]}, index=["U", "V"])
        together = transform_axes(pd.concat([companions, heldout])[train.columns[::-1]], fitted)
        pd.testing.assert_frame_equal(together.loc[["H"]], alone)
        self.assertAlmostEqual(alone.iloc[0, 0], 4. / np.std(x, ddof=1))
        pd.testing.assert_series_equal(fitted["means"], means_before)
        pd.testing.assert_series_equal(fitted["scales"], scales_before)
        pd.testing.assert_frame_equal(transform_axes(train, fitted), fitted["train_scores"])

    def test_equal_family_weights_prevent_duplicate_member_inflation(self):
        x = np.array([-2., -1., 1., 2.])
        train = pd.DataFrame({"a1": x, "a2": 2*x, "b": 3*x, "c": 4*x},
                             index=["T1", "T2", "T3", "T4"])
        metadata = pd.DataFrame({"family": ["family_a", "family_a", "family_b", "family_c"]},
                                index=train.columns)
        fitted = fit_axes(train, metadata)
        module, = fitted["module_members"]
        weights = fitted["score_weights"][module]
        np.testing.assert_allclose(weights.loc[["a1", "a2", "b", "c"]], [1/6, 1/6, 1/3, 1/3])
        family_weights = weights.groupby(metadata.family).sum()
        np.testing.assert_allclose(family_weights, [1/3, 1/3, 1/3])
        # Only family A differs from its training mean by one SD in this heldout row.
        heldout = fitted["means"].to_frame().T.copy(deep=True)
        heldout.index = ["H"]
        heldout.loc["H", ["a1", "a2"]] += fitted["scales"].loc[["a1", "a2"]]
        self.assertAlmostEqual(transform_axes(heldout, fitted).iloc[0, 0], 1/3)

    def test_two_families_do_not_create_an_eligible_three_member_axis(self):
        train = pd.DataFrame({"a1": [0., 1., 2.], "a2": [0., 2., 4.], "b": [0., 3., 6.]})
        metadata = pd.DataFrame({"family": ["a", "a", "b"]}, index=train.columns)
        fitted = fit_axes(train, metadata)
        self.assertEqual(fitted["module_members"], {})
        self.assertEqual(fitted["train_scores"].shape, (3, 0))
        self.assertEqual(transform_axes(train.iloc[[0]], fitted).shape, (1, 0))


if __name__ == "__main__":
    unittest.main()

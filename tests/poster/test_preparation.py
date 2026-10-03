"""Small-array checks for response preparation; no historical data are read."""
import unittest

import numpy as np

from poster_analysis.preparation import bin_curves, fit_representation


class PreparationContracts(unittest.TestCase):
    def test_binning_rejects_partial_curves_but_preserves_wholly_missing(self):
        curves = np.stack([np.arange(40, dtype=float), np.full(40, np.nan)])
        np.testing.assert_allclose(bin_curves(curves)[0], np.arange(2., 40., 5.))
        self.assertTrue(np.isnan(bin_curves(curves)[1]).all())
        partial = curves.copy()
        partial[0, 12] = np.nan
        with self.assertRaisesRegex(ValueError, 'Partial raw curves'):
            bin_curves(partial)

    def test_missing_limited_suppressed_and_measured_zero_remain_distinct(self):
        raw = np.full((2, 4, 2, 40), np.nan)
        raw[0, 1, 0] = 1.                 # Only one observed animal.
        raw[:, 2, 0] = [[1.] * 40, [-1.] * 40]  # Two animals, zero coherent mean.
        raw[:, 3, 0] = 2.                 # Stable response identifies cell 0's template.
        raw[:, 0, 1] = 0.                 # Measured zero cannot identify cell 1's shape.
        result = fit_representation(raw, None, ['A', 'B', 'C', 'D'], threshold=.5)
        np.testing.assert_array_equal(result['status'][:, 0],
                                      ['missing', 'limited_n', 'below_snr', 'retained'])
        self.assertTrue(np.isnan(result['coefficients'][:2, 0]).all())
        self.assertTrue(np.isnan(result['reconstruction'][:2, 0]).all())
        self.assertEqual(result['coefficients'][2, 0], 0.)
        np.testing.assert_array_equal(result['reconstruction'][2, 0], np.zeros(8))
        self.assertGreater(result['coefficients'][3, 0], 0.)
        # A zero passes an ungated fit but remains distinguishable from suppression.
        ungated = fit_representation(raw, None, ['A', 'B', 'C', 'D'], threshold=None)
        self.assertEqual(ungated['status'][0, 1], 'observed_zero')
        self.assertEqual(ungated['coefficients'][0, 1], 0.)
        np.testing.assert_array_equal(ungated['reconstruction'][0, 1], np.zeros(8))
        self.assertFalse(ungated['template_identified'][1])
        self.assertTrue(np.isnan(ungated['templates'][1]).all())
        self.assertTrue(np.isnan(raw[:, 0, 0]).all())  # Inputs are not zero-filled in place.

    def test_duplicate_condition_keeps_each_strain_equal_weight(self):
        means = np.array([[2., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 3., 0., 0., 0., 0., 0.]])
        curves = np.repeat(means, 5, axis=1)
        raw = np.repeat(curves[None, :, None, :], 2, axis=0)
        reference = fit_representation(raw, None, ['A', 'B'], threshold=None)
        repeated = raw[:, [0, 0, 1]]
        same_strain = fit_representation(repeated, None, ['A', 'A', 'B'], threshold=None)
        np.testing.assert_allclose(same_strain['templates'], reference['templates'],
                                   atol=1e-12, rtol=1e-12)
        # If the duplicate is a distinct strain it legitimately gains another vote.
        distinct_strain = fit_representation(repeated, None, ['A', 'C', 'B'], threshold=None)
        self.assertGreater(np.max(np.abs(distinct_strain['templates']
                                         - reference['templates'])), .1)
        self.assertAlmostEqual(float(np.mean(reference['templates'] ** 2)), 1.)
        self.assertGreater(reference['templates'][0, np.argmax(np.abs(reference['templates'][0]))], 0.)

    def test_snr_cutoff_refits_template_instead_of_only_zeroing_coefficients(self):
        bins = np.zeros((2, 2, 1, 8))
        bins[:, 0, 0, 0] = 1.  # Reproducible first-bin response in both animals.
        bins[1, 1, 0, 1] = 6.  # Larger second-bin mean, but zero across-animal SNR.
        raw = np.repeat(bins, 5, axis=-1)
        ungated = fit_representation(raw, None, ['A', 'B'], threshold=None)
        gated = fit_representation(raw, None, ['A', 'B'], threshold=.5)
        self.assertEqual(int(np.argmax(np.abs(ungated['templates'][0]))), 1)
        self.assertEqual(int(np.argmax(np.abs(gated['templates'][0]))), 0)
        self.assertEqual(ungated['template_n_conditions'][0], 2)
        self.assertEqual(gated['template_n_conditions'][0], 1)
        self.assertEqual(gated['status'][1, 0], 'below_snr')
        self.assertEqual(gated['coefficients'][1, 0], 0.)
        # The reliable first-bin signal must now be reconstructed exactly.
        np.testing.assert_allclose(gated['reconstruction'][0, 0], bins[0, 0, 0], atol=1e-12)
        self.assertGreater(float(np.sum((ungated['reconstruction'][0, 0]
                                        - bins[0, 0, 0]) ** 2)), .9)


if __name__ == '__main__':
    unittest.main()

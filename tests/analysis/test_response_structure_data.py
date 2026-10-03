"""Focused checks of the pre-existing scientific aggregation convention."""
import unittest

import numpy as np
import pandas as pd

from bacteria_analysis.response_structure_data import CLASSES, _prepare_table


def raw_row(value, *, date="20260101", worm="w1", trial=0, point=5,
            neuron="ASKL", stimulus="A001 stationary"):
    return dict(date=date, worm_key=worm, segment_index=trial, time_point=point,
                neuron=neuron, stim_name=stimulus, delta_F_over_F0=value,
                start_time=5, end_time=15)


class PreparationTests(unittest.TestCase):
    def test_bilateral_then_trials_per_volume_then_bins_with_missingness(self):
        rows = [raw_row(0), raw_row(4, neuron="ASKR"),  # Trial 0 / volume 5 -> 2
                raw_row(10, trial=1),                  # Trial 1 / volume 5 -> 10
                raw_row(np.nan, trial=1, neuron="ASKR"),
                raw_row(0, point=6), raw_row(np.nan, point=6, trial=1)]
        out = _prepare_table(pd.DataFrame(rows))
        row = out["observations"].query("neuron_class == 'ASK' and bin_index == 0").iloc[0]
        # Volume means 6 and 0 -> 3. Averaging each trial first would give 5.5.
        self.assertEqual(row.response, 3)
        self.assertEqual(row.n_volumes, 2)
        self.assertEqual(len(out["observations"]), len(CLASSES) * 8)

    def test_single_side_zero_and_nonfinite_are_distinct(self):
        out = _prepare_table(pd.DataFrame([
            raw_row(0), raw_row(np.inf, point=6), raw_row(-np.inf, point=7),
            raw_row(np.nan, point=8), raw_row(2, neuron="ASER"),
        ]))
        obs = out["observations"]
        ask = obs.query("neuron_class == 'ASK' and bin_index == 0").iloc[0]
        self.assertEqual(ask.response, 0)
        self.assertEqual(ask.n_volumes, 1)
        self.assertEqual(obs.query("neuron_class == 'ASER' and bin_index == 0").iloc[0].response, 2)
        self.assertTrue(obs.query("neuron_class == 'ASEL'").response.isna().all())
        cov = out["coverage"]
        self.assertEqual(cov.query("neuron_class == 'ASK' and bin_index == 0").iloc[0].n_animals, 1)
        self.assertEqual(cov.query("neuron_class == 'ASK' and bin_index == 1").iloc[0].n_animals, 0)

    def test_animal_identity_does_not_pool_dates(self):
        out = _prepare_table(pd.DataFrame([raw_row(1), raw_row(9, date="20260102")]))
        rows = out["observations"].query("neuron_class == 'ASK' and bin_index == 0")
        self.assertEqual(rows.animal_id.tolist(), ["20260101|w1", "20260102|w1"])
        self.assertEqual(rows.response.tolist(), [1, 9])
        self.assertEqual(out["metadata"]["n_conditions"], 2)
        self.assertEqual(len(out["coverage"]), 2 * len(CLASSES) * 8)

    def test_unrecognized_names_and_ambiguous_suffixes_are_reported(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized stimulus names"):
            _prepare_table(pd.DataFrame([raw_row(1, stimulus="control")]))
        with self.assertRaisesRegex(ValueError, "explicit block mapping"):
            _prepare_table(pd.DataFrame([raw_row(1), raw_row(2, stimulus="A001 second")]))

    def test_all_nan_trace_and_excluded_only_condition_remain_in_audit(self):
        out = _prepare_table(pd.DataFrame([
            raw_row(np.nan), raw_row(10, neuron="AFDL", stimulus="A002 stationary"),
        ]))
        self.assertTrue(out["observations"].response.isna().all())
        self.assertEqual(out["coverage"].n_animals.sum(), 0)
        self.assertEqual(len(out["coverage"]), 2 * len(CLASSES) * 8)
        self.assertEqual(out["trial_coverage"].n_observed_volumes.sum(), 0)


if __name__ == "__main__":
    unittest.main()

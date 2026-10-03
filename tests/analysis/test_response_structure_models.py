"""Scientific invariants for the shared-template animal prediction analysis."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bacteria_analysis.response_structure import (feature_weights, fit_models, leave_one_animal_out, _rank_one,
                                summarize_predictions, evidence_fingerprints, _top_three_refit)


class ResponseStructureModelsTests(unittest.TestCase):
    def test_evidence_fingerprint_binds_curves_not_verification_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'tables').mkdir(); (root/'data').mkdir()
            for name in ['predictions.parquet', 'templates.csv', 'coefficients.csv',
                         'template_stability.csv', 'top3_exclusions.csv']:
                (root/'tables'/name).write_bytes(b'original evidence')
            (root/'analysis_parameters.json').write_text('{}')
            path = root/'data/preparation_metadata.json'
            path.write_text(json.dumps({'raw_sha256': 'raw'}))
            before = evidence_fingerprints(root)
            path.write_text(json.dumps({'raw_sha256': 'raw', 'verification': 'new independent audit'}))
            self.assertEqual(before, evidence_fingerprints(root))
            (root/'tables/predictions.parquet').write_bytes(b'changed local curves')
            self.assertNotEqual(before, evidence_fingerprints(root))

    def test_top_three_remove_whole_strains_then_reestimate_all_coefficients(self):
        h = np.array([1., 2., .5])
        a = np.array([10., 9., 8., 7., 2., 1.])
        y = a[:, None, None] * h[None,None,:]
        n = np.full(y.shape, 3)
        strains = ['a','a','b','c','d','e']
        full = fit_models(y, n, strains)
        prediction, _, records = _top_three_refit(y, n, strains, full)
        self.assertEqual([r['sample_id'] for r in records], ['a','b','c'])
        np.testing.assert_allclose(prediction, y, atol=1e-10)

    def test_prediction_summary_does_not_weight_by_animal_or_block_counts(self):
        records = []
        for strain, block, value, count in [("a", "d1", 1., 6), ("a", "d2", 3., 3), ("b", "d3", 2., 4)]:
            for i in range(count):
                for cell in ["c1", "c2"]:
                    for t in range(2):
                        records.append(dict(window="0-10s", strain=strain, block=block,
                            cell=cell, bin_index=t, animal_id=f"{block}|{i}", actual=value,
                            eligible=True, pred_B=0., pred_M0=0., pred_M1=0., pred_M2=0.))
        summary = summarize_predictions(pd.DataFrame(records))
        self.assertAlmostEqual(summary["overall_summary"].mse_M1.iloc[0], 4.5)

    def test_full_time_reference_recovers_non_rank_one_mean(self):
        y = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])
        fit = fit_models(y, np.full(y.shape, 3), list("abc"))
        self.assertEqual(fit["losses"]["M2"], 0.)
        self.assertGreater(fit["losses"]["M1"], .1)

    def test_disconnected_time_support_is_not_an_identified_template(self):
        y = np.array([[1., np.nan], [2., np.nan], [np.nan, 10.], [np.nan, 20.]])
        w = np.isfinite(y).astype(float)
        _, _, _, diagnostic = _rank_one(y, w)
        self.assertFalse(diagnostic["identified"])

    def test_weights_equal_strains_then_blocks_and_cells(self):
        mask = np.ones((3, 2, 4), dtype=bool)
        w = feature_weights(mask, ["A001", "A001", "A002"])
        self.assertAlmostEqual(w.sum(), 1)
        np.testing.assert_allclose(w.sum(axis=(1, 2)), [.25, .25, .5])
        np.testing.assert_allclose(w.sum(axis=(0, 2)), [.5, .5])

    def test_signed_shared_shape_and_nonnegative_gain(self):
        h = np.array([[1., 2., .5, -.5], [.5, -1., 2., 1.]])
        a = np.array([[1., 1.], [-2., .2], [.2, -2.], [2., .5]])
        y = a[:, :, None] * h[None]
        fit = fit_models(y, np.full(y.shape, 3), list("abcd"))
        np.testing.assert_allclose(fit["predictions"]["M1"], y, atol=1e-7)
        self.assertTrue((fit["gains"] >= 0).all())
        np.testing.assert_allclose(np.mean(fit["templates"] ** 2, axis=1), 1)
        for row in fit["templates"]:
            self.assertGreater(row[np.argmax(abs(row))], 0)
        loss = fit["losses"]
        self.assertGreater(loss["M0"], loss["M1"] + .01)
        self.assertGreaterEqual(loss["B"] + 1e-10, loss["M0"])
        self.assertGreaterEqual(loss["M1"] + 1e-10, loss["M2"])

    def test_two_training_animals_required_per_window(self):
        y = np.array([[[1., 2., np.nan, 0.]], [[3., 4., 5., 0.]]])
        n = np.array([[[3, 1, 0, 3]], [[3, 3, 3, 3]]])
        fit = fit_models(y, n, ["a", "b"])
        self.assertFalse(fit["mask"][0, 0, 1])
        self.assertTrue(fit["mask"][0, 0, 3])
        self.assertTrue(np.isnan(fit["predictions"]["M2"][0, 0, 1]))
        self.assertEqual(fit["predictions"]["M2"][0, 0, 3], 0)

    def test_heldout_values_cannot_change_their_own_predictions(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(4, 5, 2, 4))
        first = leave_one_animal_out(x, list("abcde"), stress=False)
        changed = x.copy()
        changed[0] += 100  # All conditions and cells of one animal together.
        second = leave_one_animal_out(changed, list("abcde"), stress=False)
        for model in ("B", "M0", "M1", "M2"):
            np.testing.assert_allclose(first["predictions"][model][0],
                                       second["predictions"][model][0], atol=1e-10)
        np.testing.assert_allclose(first["predictions"]["M2"][0], x[1:].mean(axis=0))

    def test_training_one_never_scored_and_zero_is_observed(self):
        x = np.zeros((3, 2, 1, 3))
        x[2, 1] = np.nan
        out = leave_one_animal_out(x, ["a", "b"], stress=False)
        self.assertTrue(out["score_mask"][:, 0].all())
        self.assertFalse(out["score_mask"][:, 1].any())
        self.assertTrue(np.isfinite(out["predictions"]["M1"][:, 0]).all())


if __name__ == "__main__":
    unittest.main()

"""Focused profile-control regressions; no scientific data or full fits."""
import pickle
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bacteria_analysis import hmds_refinement
from bacteria_analysis import neural_hmds


class ProfileRidgeTests(unittest.TestCase):
    def run_profile(self, checkpoint, fallback_converges, budget=6, slow=True,
                    anchor_only=False, other_gradient=0.):
        reference = dict(coordinates=pd.DataFrame(np.zeros((3, 2))), parameters={},
                         optimizer_result=SimpleNamespace(x=np.r_[np.zeros(4), np.log(20.), np.zeros(3)]))
        profile_calls = 0
        anchor_changed = False

        def reanchor(ref, x, *args, **kwargs):
            nonlocal anchor_changed
            anchor_changed |= kwargs.get('avoid_current_anchor', False)
            ref['optimizer_result'].x = x.copy()
            return ref

        def refine(ref, *args, **kw):
            nonlocal profile_calls
            initial = kw['fixed_lambda'] < 21
            converged = (initial or not slow or (anchor_only and anchor_changed)
                         or (fallback_converges and kw['ridge_fraction'] < 1e-7))
            blocks = 1 if converged else kw['max_blocks']
            x = kw['initial_parameters'].copy()
            if not initial:
                # A retry must continue the failed fit, not restart at the base.
                self.assertEqual(x[0], profile_calls)
                x[0] += 1
                profile_calls += 1
            x[4] = np.log(kw['fixed_lambda'])
            self.assertEqual(kw['gradient_tolerance'], 1e-7)
            self.assertEqual(kw['gradient_metric'], 'local_geometric')
            return dict(parameters_vector=x, settings={},
                        history=pd.DataFrame({'n_iter': [kw['iterations_per_block']] * blocks}),
                        diagnostics=dict(gradient_converged=converged,
                                         projected_grad_inf=1e-8 if converged else 1e-4,
                                         geometric_projected_grad_max=1e-8 if converged else 1e-4,
                                         anchor_geometric_grad_norm=1e-4 if anchor_only and not converged else 0.,
                                         free_coordinate_geometric_grad_max=(1e-8 if converged else
                                             max(1e-8, other_gradient) if anchor_only else 1e-4),
                                         variance_projected_grad_inf=0. if converged else other_gradient,
                                         convergence_grad_norm=1e-8 if converged else 1e-4,
                                         gradient_metric='local_geometric',
                                         d_loss_d_log_lambda=-.1 if initial else 0.,
                                         stop_reason='converged' if converged else 'block budget exhausted'))

        with patch.object(hmds_refinement, 'reanchor_hmds', side_effect=reanchor), \
                patch.object(hmds_refinement, 'refine_hmds', side_effect=refine) as mocked:
            self.calls = mocked
            return neural_hmds._profile_neural_hmds_2d(reference, checkpoint, budget, 2000, 1e-6, 1e-7)

    def test_slow_profile_reconditions_without_relaxing_tolerance(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint.pkl'
            _, fit = self.run_profile(checkpoint, fallback_converges=True)
            self.assertTrue(fit['diagnostics']['joint_gradient_converged'])
            self.assertEqual(len(fit['history']), 6)
            np.testing.assert_allclose(fit['history'].ridge_fraction, [1e-7] * 5 + [1e-10])
            state = pickle.loads(checkpoint.read_bytes())
            np.testing.assert_allclose([r['ridge_fraction'] for r in state['records']],
                                       [1e-7, 1e-7, 1e-10])

    def test_converged_profile_keeps_original_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            _, fit = self.run_profile(Path(folder) / 'checkpoint.pkl', False, slow=False)
            self.assertTrue(fit['diagnostics']['joint_gradient_converged'])
            self.assertEqual(self.calls.call_count, 2)
            np.testing.assert_allclose(fit['profile_history'].ridge_fraction, [1e-7, 1e-7])

    def test_anchor_only_failure_retries_without_ignoring_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            _, fit = self.run_profile(Path(folder) / 'checkpoint.pkl', False, anchor_only=True)
            self.assertTrue(fit['diagnostics']['joint_gradient_converged'])
            self.assertEqual(self.calls.call_count, 3)
            self.assertEqual(len(fit['history']), 6)

    def test_dominant_anchor_retries_before_other_groups_converge(self):
        # A pinned anchor can dominate while free-coordinate and variance
        # gradients still fluctuate above tolerance. Waiting for both to pass
        # can consume every block before the anchor is allowed to change.
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint.pkl'
            _, fit = self.run_profile(checkpoint, False, anchor_only=True, other_gradient=1e-6)
            self.assertTrue(fit['diagnostics']['joint_gradient_converged'])
            self.assertEqual(len(fit['history']), 6)
            state = pickle.loads(checkpoint.read_bytes())
            self.assertFalse(state['records'][-2]['gradient_converged'])
            self.assertGreater(state['records'][-2]['variance_projected_grad_inf'], 1e-7)

    def test_failed_or_unavailable_fallback_respects_budget(self):
        for budget, expected_calls in [(5, 2), (6, 3)]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / 'checkpoint.pkl'
                with self.assertRaises(neural_hmds._HMDSProfileError):
                    self.run_profile(checkpoint, fallback_converges=False, budget=budget)
                state = pickle.loads(checkpoint.read_bytes())
                self.assertFalse(state['current']['fit']['diagnostics']['gradient_converged'])
                self.assertEqual(len(state['current']['fit']['history']), budget)
                self.assertEqual(self.calls.call_count, expected_calls)


class ReanchorPrecisionTests(unittest.TestCase):
    def test_nonfinite_distance_check_cannot_pass(self):
        reference = dict(coordinates=pd.DataFrame(np.zeros((3, 2))),
                         parameters=dict(dimension=2))
        with patch.object(neural_hmds, 'hmds_geometry', side_effect=[
                (np.ones(3), None, None), (np.full(3, np.nan), None, None)]):
            with self.assertRaises(FloatingPointError):
                hmds_refinement.reanchor_hmds(
                    reference, np.zeros(8), neural_hmds.hmds_geometry, neural_hmds.hmds_loss_gradient)

    def test_rotation_between_grid_angles_preserves_likelihood(self):
        # The best angle is about -0.00034 rad: the old 0.00153-rad grid
        # chooses zero and leaves one height at 5e8. Check both supported dims.
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                chart = np.zeros((4, dimension))
                chart[:, 0] = [0., -500., 500., 0.]
                chart[:, -1] = np.log([1., 5e-6, 1e-5, 5e8])
                ids = pd.Index(['anchor', 'left', 'right', 'high'])
                pi, pj = np.triu_indices(4, 1)
                distances = neural_hmds.hmds_geometry(chart, pi, pj)[0]
                observed = distances / 23. + .01
                variance = np.full(len(pi), .02)
                x = np.r_[chart[1:].ravel(), np.log(23.), [.01, .02, .03, .04]]
                reference = dict(
                    coordinates=pd.DataFrame(neural_hmds.hmds_chart_to_ball(chart), index=ids),
                    parameters=dict(dimension=dimension, variance_floor=1e-10, variance_parameter_scale=1.),
                    optimizer_result=SimpleNamespace(x=x),
                    pairs=pd.DataFrame(dict(sample_i=ids[pi], sample_j=ids[pj],
                                            input_distance=observed, bootstrap_variance=variance)),
                )
                old_loss = neural_hmds.hmds_loss_gradient(
                    x, 4, dimension, pi, pj, observed, variance)[0]
                result = hmds_refinement.reanchor_hmds(
                    reference, x, neural_hmds.hmds_geometry, neural_hmds.hmds_loss_gradient)
                self.assertGreater(result['halfspace_coordinates'][:, -1].min(), np.log(6e-6))
                self.assertLess(result['halfspace_coordinates'][:, -1].max(), 1.)
                order = result['coordinates'].index.get_indexer(ids)
                new_distances = neural_hmds.hmds_geometry(
                    result['halfspace_coordinates'][order], pi, pj)[0]
                np.testing.assert_allclose(new_distances, distances, rtol=0, atol=1e-8)
                self.assertAlmostEqual(result['optimizer_result'].fun, old_loss, places=10)
                pd.testing.assert_frame_equal(result['pairs'], reference['pairs'])
                np.testing.assert_allclose(result['sigma'].loc[ids] ** 2, x[-4:])
                np.testing.assert_array_equal(reference['optimizer_result'].x, x)


class DisplayCenteringTests(unittest.TestCase):
    def test_centering_allows_predicted_float64_roundoff_in_tight_cluster(self):
        chart = np.array([[0., 0.], [1800., np.log(6e-7)],
                          [1800. + 2e-6, np.log(8e-7)],
                          [1800. + 4e-6, np.log(4e-7)],
                          [-8000., -8.], [36000., -12.]])
        original = chart.copy()
        origin = np.median(chart[:, 0])
        units = max(np.std(chart[:, 0]), 1.)
        # Isolate the display transformation from the center optimizer. This
        # center moves a tight cluster to larger x and loses several low bits.
        center_result = SimpleNamespace(x=np.array([(-7908. - origin) / units, 5.25]))
        with patch('scipy.optimize.minimize', return_value=center_result):
            points, error = neural_hmds.center_hmds_for_display(chart)
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue((np.linalg.norm(points, axis=1) <= 1.).all())
        self.assertLess(error, 1e-5)
        np.testing.assert_array_equal(chart, original)

    def test_centering_still_rejects_a_real_distance_change(self):
        chart = np.array([[0., 0.], [1., .2], [2., -.3], [-1., .5]])
        geometry = neural_hmds.hmds_geometry

        def changed_distances(values, ii, jj):
            distance, gi, gj = geometry(values, ii, jj)
            # Optimizer calls add a center point. Corrupt only the subsequent
            # distance check on the transformed chart, not its original.
            if len(values) == len(chart) and not np.array_equal(values, chart):
                distance = distance + .01
            return distance, gi, gj

        with patch.object(neural_hmds, 'hmds_geometry', side_effect=changed_distances):
            with self.assertRaisesRegex(RuntimeError, 'Display centering changed'):
                neural_hmds.center_hmds_for_display(chart)


class HorizontalDifferenceTests(unittest.TestCase):
    def problem(self):
        # Small heights at a large horizontal offset leave a narrow FD window:
        # large steps truncate, while tiny steps fall below 32 float64 ulps.
        chart = np.array([[0., 0.], [1800., np.log(6e-7)],
                          [1800. + 2e-6, np.log(8e-7)],
                          [1800. + 4e-6, np.log(4e-7)]])
        ids = pd.Index(['anchor', 'b', 'c', 'd'])
        pi, pj = np.triu_indices(4, 1)
        observed = neural_hmds.hmds_geometry(chart, pi, pj)[0] / 28.
        variance = np.full(len(pi), 10.)
        x = np.r_[chart[1:].ravel(), np.log(28.), np.zeros(4)]
        loss, grad = neural_hmds.hmds_loss_gradient(x, 4, 2, pi, pj, observed, variance)
        ref = dict(coordinates=pd.DataFrame(chart, index=ids),
                   parameters=dict(dimension=2, variance_floor=1e-10, variance_parameter_scale=1.,
                                   coordinate_system='upper half-space: x and log-height'),
                   optimizer_result=SimpleNamespace(x=x, fun=loss, jac=grad),
                   pairs=pd.DataFrame(dict(sample_i=ids[pi], sample_j=ids[pj],
                                           input_distance=observed, bootstrap_variance=variance)))
        return ref, dict(parameters_vector=x, settings=dict(gradient_metric='local_geometric'))

    def test_narrow_precision_window_passes_without_relaxing_tolerances(self):
        ref, fit = self.problem()
        with tempfile.TemporaryDirectory() as directory:
            report = neural_hmds.check_neural_hmds_fit(ref, fit, directory)
            steps = pd.read_csv(Path(directory) / 'gradient_steps.csv')
            summary = pd.read_csv(Path(directory) / 'gradient_summary.csv')
        self.assertTrue(report['all_gradient_checks_passed'])
        self.assertEqual(report['checked_parameters'], len(fit['parameters_vector']))
        self.assertTrue(summary.adjacent_steps_agree.all())
        horizontal = steps.loc[steps.group == 'horizontal']
        self.assertTrue(horizontal.note.str.contains('below float64 spacing').any())
        self.assertTrue((horizontal.loc[horizontal.agrees, 'error_over_tolerance'] <= 1.).all())

    def test_wrong_raw_derivative_is_rejected_even_if_geometric_norm_is_small(self):
        ref, fit = self.problem()
        objective = neural_hmds.hmds_loss_gradient

        def wrong_gradient(*args, **kwargs):
            loss, grad = objective(*args, **kwargs)
            grad[0] += .01  # Local geometric component is only 6e-9.
            return loss, grad

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(neural_hmds, 'hmds_loss_gradient', side_effect=wrong_gradient):
            with self.assertRaisesRegex(RuntimeError, 'numerical gradient checks failed'):
                neural_hmds.check_neural_hmds_fit(ref, fit, directory)
            summary = pd.read_csv(Path(directory) / 'gradient_summary.csv')
            self.assertFalse(summary.set_index('parameter').loc['b:x1', 'adjacent_steps_agree'])


class GeometricGradientTests(unittest.TestCase):
    def problem(self, dimension=2, exact=False):
        chart = np.zeros((5, dimension))
        chart[:, 0] = [0., 2., 3., 4., 6.]
        chart[:, -1] = [0., .2, .4, .6, .2]
        if dimension == 3:
            chart[:, 1] = [0., .5, -.7, .3, .9]
        ids = pd.Index(list('abcde'))
        pi, pj = np.triu_indices(5, 1)
        observed = neural_hmds.hmds_geometry(chart, pi, pj)[0] / 2.
        if not exact:
            observed = observed * .9 + .02
        variance = np.full(len(pi), .02)
        x = np.r_[chart[1:].ravel(), np.log(2.), np.zeros(5)]
        loss, grad = neural_hmds.hmds_loss_gradient(x, 5, dimension, pi, pj, observed, variance)
        ref = dict(coordinates=pd.DataFrame(neural_hmds.hmds_chart_to_ball(chart), index=ids),
                   parameters=dict(dimension=dimension, variance_floor=1e-10, variance_parameter_scale=1.,
                                   coordinate_system='upper half-space: x and log-height'),
                   optimizer_result=SimpleNamespace(x=x, fun=loss, jac=grad),
                   pairs=pd.DataFrame(dict(sample_i=ids[pi], sample_j=ids[pj],
                                           input_distance=observed, bootstrap_variance=variance)))
        return ref, x, grad

    def norms(self, x, grad, dimension, fixed=False, anchor=None):
        nc = 4 * dimension
        lo, hi = np.full(len(x), -np.inf), np.full(len(x), np.inf)
        lo[nc + 1:] = 0
        if fixed:
            lo[nc] = hi[nc] = x[nc]
        pg = neural_hmds.hmds_projected_gradient(x, grad, lo, hi)
        if anchor is None:  # Exact small synthetic derivatives used by the unit tests.
            raw = grad[:nc].reshape(4, dimension)
            chart = x[:nc].reshape(4, dimension)
            anchor = np.r_[-raw[:, :-1].sum(axis=0),
                           -np.sum(chart[:, :-1] * raw[:, :-1]) - raw[:, -1].sum()]
        return hmds_refinement.hmds_gradient_diagnostics(x, grad, pg, 5, dimension, anchor)

    def anchor(self, ref, x, dimension):
        ids, pairs = ref['coordinates'].index, ref['pairs']
        return hmds_refinement.hmds_anchor_gradient(
            x, len(ids), dimension, ids.get_indexer(pairs.sample_i), ids.get_indexer(pairs.sample_j),
            pairs.input_distance.to_numpy(), pairs.bootstrap_variance.to_numpy(), 1., neural_hmds.hmds_geometry)

    def test_norm_invariant_under_isometry_and_anchor_change(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                ref, x, grad = self.problem(dimension)
                original = self.norms(x, grad, dimension, anchor=self.anchor(ref, x, dimension))
                rotated = hmds_refinement.reanchor_hmds(
                    ref, x, neural_hmds.hmds_geometry, neural_hmds.hmds_loss_gradient)
                self.assertNotEqual(rotated['coordinates'].index[0], ref['coordinates'].index[0])
                rx = rotated['optimizer_result'].x
                changed = self.norms(rx, rotated['optimizer_result'].jac, dimension,
                                     anchor=self.anchor(rotated, rx, dimension))
                for key in ('coordinate_geometric_grad_max', 'geometric_projected_grad_max',
                            'variance_projected_grad_inf', 'log_lambda_grad_abs'):
                    self.assertAlmostEqual(original[key], changed[key], places=10)

    def test_orthonormal_components_match_directional_derivative(self):
        ref, x, grad = self.problem()
        pi, pj = np.triu_indices(5, 1)
        height = np.exp(x[1])
        local = grad[:2] * [height, 1.]
        unit = local / np.linalg.norm(local)
        direction = np.zeros_like(x)
        direction[:2] = unit * [height, 1.]
        def loss(z):
            return neural_hmds.hmds_loss_gradient(
                z, 5, 2, pi, pj, ref['pairs'].input_distance.to_numpy(), np.full(len(pi), .02))[0]
        numeric = (loss(x + 1e-5 * direction) - loss(x - 1e-5 * direction)) / 2e-5
        self.assertAlmostEqual(numeric, np.linalg.norm(local), places=8)

    def test_anchor_is_not_omitted(self):
        _, x, _ = self.problem()
        x[1:8:2] = -20.
        grad = np.zeros_like(x)
        grad[0] = 1e-3
        norms = self.norms(x, grad, 2, fixed=True)
        self.assertGreaterEqual(norms['anchor_geometric_grad_norm'], 1e-3)
        self.assertGreaterEqual(norms['geometric_projected_grad_max'], 1e-3)

    def test_unrepresentable_height_cannot_suppress_gradient(self):
        _, x, _ = self.problem()
        x[1:8:2] = -1000.
        norms = self.norms(x, np.zeros_like(x), 2)
        self.assertFalse(norms['coordinate_representation_valid'])
        self.assertEqual(norms['geometric_projected_grad_max'], np.inf)

    def test_nan_in_any_group_cannot_pass(self):
        _, x, _ = self.problem()
        for index in (0, 8, 9):
            grad = np.zeros_like(x)
            grad[index] = np.nan
            self.assertFalse(self.norms(x, grad, 2)['geometric_projected_grad_max'] <= 1e-6)

    def test_lambda_and_variance_constraints_are_retained(self):
        _, x, _ = self.problem()
        for index in (8, 9):
            grad = np.zeros_like(x)
            grad[index] = -1e-3
            self.assertEqual(self.norms(x, grad, 2)['geometric_projected_grad_max'], 1e-3)
        grad = np.zeros_like(x)
        grad[8], grad[9] = 1e-3, 1e-3  # Fixed lambda and outward gradient at q=0.
        self.assertEqual(self.norms(x, grad, 2, fixed=True)['geometric_projected_grad_max'], 0.)
        self.assertEqual(self.norms(x, grad, 2, fixed=True)['log_lambda_grad_abs'], 1e-3)

    def test_final_check_rejects_unfinished_lambda_or_variance(self):
        ref, x, _ = self.problem(exact=True)
        fit = dict(parameters_vector=x, settings=dict(gradient_metric='local_geometric'))
        for index in (8, 9):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                grad = np.zeros_like(x)
                grad[index] = -1e-3
                with patch.object(neural_hmds, 'hmds_loss_gradient', return_value=(0., grad)):
                    with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                        neural_hmds.check_neural_hmds_fit(ref, fit, directory)
                report = json.loads((Path(directory) / 'convergence.json').read_text())
                self.assertFalse(report['joint_gradient_converged'])
                self.assertEqual(report['joint_convergence_grad_norm'], 1e-3)

    def test_exact_small_fit_passes_independent_final_checks(self):
        ref, x, _ = self.problem(exact=True)
        fit = dict(parameters_vector=x, settings=dict(gradient_metric='local_geometric'))
        with tempfile.TemporaryDirectory() as directory:
            report = neural_hmds.check_neural_hmds_fit(ref, fit, directory)
        self.assertTrue(report['joint_gradient_converged'])
        self.assertTrue(report['all_gradient_checks_passed'])
        self.assertEqual(report['checked_parameters'], len(x))

    def test_clustered_points_do_not_create_a_false_anchor_gradient(self):
        rng = np.random.default_rng(0)
        chart = np.zeros((8, 2))
        chart[1:, 0] = np.exp(-45.) * rng.uniform(1., 5., 7)
        chart[1:, 1] = -45. + rng.normal(0., .3, 7)
        pi, pj = np.triu_indices(8, 1)
        observed = neural_hmds.hmds_geometry(chart, pi, pj)[0].copy()
        observed[pi > 0] += rng.normal(size=int((pi > 0).sum())) * 1e-8
        variance = np.ones(len(pi))
        x = np.r_[chart[1:].ravel(), 0., np.zeros(8)]
        _, grad = neural_hmds.hmds_loss_gradient(x, 8, 2, pi, pj, observed, variance)
        lo, hi = np.full(len(x), -np.inf), np.full(len(x), np.inf)
        lo[15:] = 0
        pg = neural_hmds.hmds_projected_gradient(x, grad, lo, hi)
        anchor = hmds_refinement.hmds_anchor_gradient(
            x, 8, 2, pi, pj, observed, variance, 1., neural_hmds.hmds_geometry)
        norms = hmds_refinement.hmds_gradient_diagnostics(x, grad, pg, 8, 2, anchor)
        np.testing.assert_array_equal(anchor, [0., 0.])
        self.assertTrue(norms['coordinate_representation_valid'])
        self.assertLess(norms['geometric_projected_grad_max'], 1e-7)
        self.assertGreater(norms['anchor_gradient_reconstruction_error'], 1e-6)


if __name__ == '__main__':
    unittest.main()

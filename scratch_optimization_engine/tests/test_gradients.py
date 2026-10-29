"""
Unit Verification & Numerical Finite-Difference Gradient Checking Suite
======================================================================

Mathematically validates all analytical gradient derivations across:
1. Multivariate Polynomial Regression loss surfaces (MSE, Ridge L2, Elastic Net).
2. Trajectory Smoothing variational objectives (Smoothness, Tension, Obstacle Potential).
3. Optimizer moment recurrences and bias corrections (SGD, RMSProp, Adam).
4. Asymptotic convergence equivalence between Closed-Form and Iterative solvers.

Finite-Difference Formulation:
------------------------------
For objective functional $f: \\mathbb{R}^d \\to \\mathbb{R}$, the central difference quotient is:
$$\\left[ \\nabla_{\\text{num}} f(x) \\right]_i = \\frac{f(x + \\epsilon e_i) - f(x - \\epsilon e_i)}{2\\epsilon}$$

By Taylor theorem:
$$f(x + \\epsilon e_i) = f(x) + \\epsilon [\\nabla f(x)]_i + \\frac{\\epsilon^2}{2} [H f(x)]_{ii} + \\frac{\\epsilon^3}{6} f'''(x)_i + O(\\epsilon^4)$$
$$f(x - \\epsilon e_i) = f(x) - \\epsilon [\\nabla f(x)]_i + \\frac{\\epsilon^2}{2} [H f(x)]_{ii} - \\frac{\\epsilon^3}{6} f'''(x)_i + O(\\epsilon^4)$$
Subtracting yields:
$$\\frac{f(x + \\epsilon e_i) - f(x - \\epsilon e_i)}{2\\epsilon} = [\\nabla f(x)]_i + \\frac{\\epsilon^2}{6} f'''(x)_i + O(\\epsilon^4) = [\\nabla f(x)]_i + O(\\epsilon^2)$$

Relative Error Metric:
----------------------
$$\\text{rel\\_error} = \\frac{\\|\\nabla_{\\text{analytical}} - \\nabla_{\\text{num}}\\|_2}{\\max(\\|\\nabla_{\\text{analytical}}\\|_2 + \\|\\nabla_{\\text{num}}\\|_2, 10^{-15})} < 10^{-7}$$
"""

from __future__ import annotations
import os
import sys
import unittest
from typing import Callable, Tuple
import numpy as np

# Ensure both parent workspace and scratch_optimization_engine directory are on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_current_dir)
_parent_dir = os.path.dirname(_engine_dir)
for _p in [_parent_dir, _engine_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scratch_optimization_engine.core.optimizers import SGD, RMSProp, Adam
    from scratch_optimization_engine.core.regression import (
        MultivariatePolynomialRegression,
        PolynomialFeatures,
        soft_thresholding,
    )
    from scratch_optimization_engine.core.path_planner import (
        OccupancyGrid,
        AStarPlanner,
        TrajectorySmoother,
    )
except (ModuleNotFoundError, ImportError):
    from core.optimizers import SGD, RMSProp, Adam
    from core.regression import (
        MultivariatePolynomialRegression,
        PolynomialFeatures,
        soft_thresholding,
    )
    from core.path_planner import (
        OccupancyGrid,
        AStarPlanner,
        TrajectorySmoother,
    )


def compute_numerical_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute central finite-difference gradient:
    $$\\nabla_{\\text{num}} f(x)_i = \\frac{f(x + \\epsilon e_i) - f(x - \\epsilon e_i)}{2\\epsilon}$$

    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Scalar objective function $f(x)$.
    x : np.ndarray
        Evaluation coordinate vector or matrix.
    eps : float
        Perturbation step size $\\epsilon > 0$ (default: 1e-6).

    Returns
    -------
    np.ndarray
        Numerical gradient tensor matching shape of `x`.

    Complexity:
    -----------
    - Time: $2 \\cdot d \\cdot O(\\text{eval}(f))$ function evaluations.
    - Space: $O(d)$ perturbation memory.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    grad_flat = grad.ravel()

    for i in range(x_flat.size):
        old_val = x_flat[i]

        x_flat[i] = old_val + eps
        f_plus = func(x)

        x_flat[i] = old_val - eps
        f_minus = func(x)

        grad_flat[i] = (f_plus - f_minus) / (2.0 * eps)
        x_flat[i] = old_val

    return grad.reshape(orig_shape)


def compute_relative_error(grad_analytic: np.ndarray, grad_numeric: np.ndarray) -> float:
    """
    Calculate relative normalized Euclidean error:
    $$\\text{rel\\_error} = \\frac{\\|g_{\\text{analytic}} - g_{\\text{numeric}}\\|_2}{\\max(\\|g_{\\text{analytic}}\\|_2 + \\|g_{\\text{numeric}}\\|_2, 10^{-15})}$$
    """
    diff_norm = float(np.linalg.norm(grad_analytic - grad_numeric))
    denom = float(np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric))
    if denom < 1e-15:
        return 0.0
    return diff_norm / denom


class TestRegressionGradients(unittest.TestCase):
    """Rigorous gradient checking for Regression Loss functions."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.n_samples = 40
        self.n_features = 4
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randn(self.n_samples)

    def test_ols_loss_gradient(self) -> None:
        """Verify Ordinary Least Squares (OLS) MSE loss gradient within 1e-7."""
        poly = PolynomialFeatures(degree=1, include_bias=True)
        Phi = poly.fit_transform(self.X)
        D = Phi.shape[1]
        theta = np.random.randn(D)

        model = MultivariatePolynomialRegression(degree=1, alpha=0.0, fit_intercept=True)
        analytical_grad = model.compute_gradient_smooth(Phi, self.y, theta)

        def loss_func(th: np.ndarray) -> float:
            return model.compute_loss(Phi, self.y, th)

        numerical_grad = compute_numerical_gradient(loss_func, theta, eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"OLS gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )

    def test_ridge_loss_gradient(self) -> None:
        """Verify Ridge Regularized (L2) loss gradient within 1e-7."""
        poly = PolynomialFeatures(degree=2, include_bias=True)
        Phi = poly.fit_transform(self.X)
        D = Phi.shape[1]
        theta = np.random.randn(D)

        model = MultivariatePolynomialRegression(degree=2, alpha=0.5, l1_ratio=0.0, fit_intercept=True)
        analytical_grad = model.compute_gradient_smooth(Phi, self.y, theta)

        def loss_func(th: np.ndarray) -> float:
            return model.compute_loss(Phi, self.y, th)

        numerical_grad = compute_numerical_gradient(loss_func, theta, eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"Ridge gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )

    def test_elastic_net_smooth_gradient(self) -> None:
        """Verify ElasticNet smooth component gradient (with partial L2) within 1e-7."""
        poly = PolynomialFeatures(degree=1, include_bias=True)
        Phi = poly.fit_transform(self.X)
        D = Phi.shape[1]
        theta = np.random.randn(D)

        # alpha=0.4, l1_ratio=0.3 -> L2 weight is 0.4 * 0.7 = 0.28
        model = MultivariatePolynomialRegression(degree=1, alpha=0.4, l1_ratio=0.3, fit_intercept=True)
        analytical_grad = model.compute_gradient_smooth(Phi, self.y, theta)

        # Smooth loss: 0.5 * MSE + 0.5 * alpha * (1 - l1_ratio) * ||theta_reg||^2
        def smooth_loss_func(th: np.ndarray) -> float:
            res = Phi @ th - self.y
            mse = 0.5 * np.mean(res ** 2)
            l2 = 0.5 * 0.4 * (1.0 - 0.3) * np.sum(th[1:] ** 2)
            return float(mse + l2)

        numerical_grad = compute_numerical_gradient(smooth_loss_func, theta, eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"Elastic Net smooth gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )


class TestTrajectorySmoothingGradients(unittest.TestCase):
    """Rigorous gradient checking for Variational Trajectory Smoothing objectives."""

    def setUp(self) -> None:
        np.random.seed(1337)
        self.grid = OccupancyGrid(width=30, height=30, resolution=1.0)
        # Add single circular obstacle at center
        self.grid.add_circular_obstacle(center_x=15.0, center_y=15.0, radius=3.0)

        # Generate smooth synthetic path passing close to obstacle (with slight asymmetry to avoid Voronoi ridges)
        N = 25
        t = np.linspace(0.0, 1.0, N)
        xs = 5.2 + 19.6 * t
        ys = 4.8 + 20.4 * t + 3.2 * np.sin(np.pi * t)
        self.waypoints = np.column_stack([xs, ys])

    def test_smoothness_gradient(self) -> None:
        """Verify discrete biharmonic smoothness gradient within 1e-7."""
        smoother = TrajectorySmoother(
            grid_map=self.grid,
            w_smooth=2.0,
            w_curv=0.0,
            w_obs=0.0,
            w_orig=0.0,
        )

        P = self.waypoints.copy()
        _, analytical_grad = smoother.compute_objective_and_gradient(P, P)

        # Objective w.r.t interior waypoints
        def obj_func(interior: np.ndarray) -> float:
            full_P = P.copy()
            full_P[1:-1] = interior
            loss, _ = smoother.compute_objective_and_gradient(full_P, P)
            return loss

        numerical_grad = compute_numerical_gradient(obj_func, P[1:-1], eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"Trajectory smoothness gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )

    def test_tension_gradient(self) -> None:
        """Verify discrete tension (arc-length) gradient within 1e-7."""
        smoother = TrajectorySmoother(
            grid_map=self.grid,
            w_smooth=0.0,
            w_curv=1.5,
            w_obs=0.0,
            w_orig=0.0,
        )

        P = self.waypoints.copy()
        _, analytical_grad = smoother.compute_objective_and_gradient(P, P)

        def obj_func(interior: np.ndarray) -> float:
            full_P = P.copy()
            full_P[1:-1] = interior
            loss, _ = smoother.compute_objective_and_gradient(full_P, P)
            return loss

        numerical_grad = compute_numerical_gradient(obj_func, P[1:-1], eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"Trajectory tension gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )

    def test_obstacle_repulsion_gradient(self) -> None:
        """Verify obstacle potential field analytical gradient within 1e-7."""
        # Setup path points strictly inside the repulsive zone d < d_safe
        # and slightly offset from symmetry lines to avoid measure-zero non-differentiable Voronoi ridges
        test_pts = np.array([
            [15.2, 19.5],
            [19.3, 15.2],
            [11.1, 15.2],
        ])

        P = np.vstack([[5.0, 5.0], test_pts, [25.0, 25.0]])

        smoother = TrajectorySmoother(
            grid_map=self.grid,
            w_smooth=0.0,
            w_curv=0.0,
            w_obs=2.5,
            w_orig=0.0,
            d_safe=5.0,
        )

        _, analytical_grad = smoother.compute_objective_and_gradient(P, P)

        def obj_func(interior: np.ndarray) -> float:
            full_P = P.copy()
            full_P[1:-1] = interior
            loss, _ = smoother.compute_objective_and_gradient(full_P, P)
            return loss

        numerical_grad = compute_numerical_gradient(obj_func, P[1:-1], eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"Obstacle repulsion gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )

    def test_composite_trajectory_gradient(self) -> None:
        """Verify composite multi-objective trajectory gradient within 1e-7."""
        smoother = TrajectorySmoother(
            grid_map=self.grid,
            w_smooth=1.2,
            w_curv=0.8,
            w_obs=1.5,
            w_orig=0.3,
            d_safe=5.0,
        )

        test_pts = np.array([
            [15.2, 19.5],
            [19.3, 15.2],
            [11.1, 15.2],
        ])
        P = np.vstack([[5.0, 5.0], test_pts, [25.0, 25.0]])

        _, analytical_grad = smoother.compute_objective_and_gradient(P, P)

        def obj_func(interior: np.ndarray) -> float:
            full_P = P.copy()
            full_P[1:-1] = interior
            loss, _ = smoother.compute_objective_and_gradient(full_P, P)
            return loss

        numerical_grad = compute_numerical_gradient(obj_func, P[1:-1], eps=1e-6)
        rel_error = compute_relative_error(analytical_grad, numerical_grad)

        self.assertLess(
            rel_error,
            1e-7,
            f"Composite trajectory gradient relative error {rel_error:.3e} exceeds tolerance 1e-7",
        )


class TestOptimizerMechanics(unittest.TestCase):
    """Unit tests verifying optimizer recurrences and mathematical identities."""

    def test_adam_moment_recurrence_and_bias_corrections(self) -> None:
        """Verify Adam first & second moment updates and analytical bias correction."""
        lr = 0.01
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        adam = Adam(learning_rate=lr, betas=(beta1, beta2), eps=eps)

        params = np.array([1.0, -2.0, 0.5])
        grads = np.array([0.1, -0.4, 0.05])

        # Step 1
        updated = adam.step(params, grads)
        state = adam.state_dict()["state"]

        expected_m1 = (1.0 - beta1) * grads
        expected_v1 = (1.0 - beta2) * (grads ** 2)

        np.testing.assert_allclose(state["m"], expected_m1, rtol=1e-12)
        np.testing.assert_allclose(state["v"], expected_v1, rtol=1e-12)

        m_hat1 = expected_m1 / (1.0 - beta1)
        v_hat1 = expected_v1 / (1.0 - beta2)
        expected_params1 = params - lr * (m_hat1 / (np.sqrt(v_hat1) + eps))
        np.testing.assert_allclose(updated, expected_params1, rtol=1e-12)

        # Step 2
        grads2 = np.array([0.2, 0.1, -0.3])
        updated2 = adam.step(updated, grads2)
        state2 = adam.state_dict()["state"]

        expected_m2 = beta1 * expected_m1 + (1.0 - beta1) * grads2
        expected_v2 = beta2 * expected_v1 + (1.0 - beta2) * (grads2 ** 2)

        np.testing.assert_allclose(state2["m"], expected_m2, rtol=1e-12)
        np.testing.assert_allclose(state2["v"], expected_v2, rtol=1e-12)

        m_hat2 = expected_m2 / (1.0 - beta1 ** 2)
        v_hat2 = expected_v2 / (1.0 - beta2 ** 2)
        expected_params2 = updated - lr * (m_hat2 / (np.sqrt(v_hat2) + eps))
        np.testing.assert_allclose(updated2, expected_params2, rtol=1e-12)

    def test_rmsprop_updates(self) -> None:
        """Verify RMSProp exponentially weighted moving average updates."""
        lr = 0.005
        alpha = 0.95
        eps = 1e-8
        rms = RMSProp(learning_rate=lr, alpha=alpha, eps=eps)

        params = np.array([2.0, -1.0])
        grads = np.array([0.3, 0.8])

        updated = rms.step(params, grads)
        state = rms.state_dict()["state"]

        expected_v = (1.0 - alpha) * (grads ** 2)
        np.testing.assert_allclose(state["square_avg"], expected_v, rtol=1e-12)

        expected_params = params - lr * (grads / (np.sqrt(expected_v) + eps))
        np.testing.assert_allclose(updated, expected_params, rtol=1e-12)

    def test_sgd_momentum_updates(self) -> None:
        """Verify Polyak momentum velocity recursion."""
        lr = 0.1
        momentum = 0.9
        sgd = SGD(learning_rate=lr, momentum=momentum)

        params = np.array([10.0, -5.0])
        grads1 = np.array([1.0, 2.0])

        updated1 = sgd.step(params, grads1)
        state1 = sgd.state_dict()["state"]

        expected_v1 = grads1
        np.testing.assert_allclose(state1["velocity"], expected_v1, rtol=1e-12)
        expected_params1 = params - lr * expected_v1
        np.testing.assert_allclose(updated1, expected_params1, rtol=1e-12)

        grads2 = np.array([0.5, -1.0])
        updated2 = sgd.step(updated1, grads2)
        state2 = sgd.state_dict()["state"]

        expected_v2 = momentum * expected_v1 + grads2
        np.testing.assert_allclose(state2["velocity"], expected_v2, rtol=1e-12)
        expected_params2 = updated1 - lr * expected_v2
        np.testing.assert_allclose(updated2, expected_params2, rtol=1e-12)


class TestConvergenceEquivalence(unittest.TestCase):
    """Verify asymptotic equivalence of Closed-Form, Gradient Descent, and Coordinate Descent."""

    def test_closed_form_vs_gradient_descent_convergence(self) -> None:
        """Verify Adam gradient descent matches closed-form Normal Equation within 1e-4."""
        np.random.seed(99)
        X = np.random.randn(80, 3)
        y = 2.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * (X[:, 2] ** 2)

        # Fit Closed-Form
        model_cf = MultivariatePolynomialRegression(degree=2, alpha=0.01, solver="closed_form")
        model_cf.fit(X, y)

        # Fit Gradient Descent with Adam
        model_gd = MultivariatePolynomialRegression(
            degree=2,
            alpha=0.01,
            solver="gradient_descent",
            optimizer="adam",
            learning_rate=0.08,
            max_iter=3500,
            tol=1e-8,
        )
        model_gd.fit(X, y)

        cf_theta = model_cf.theta_
        gd_theta = model_gd.theta_

        max_discrepancy = float(np.max(np.abs(cf_theta - gd_theta)))
        self.assertLess(
            max_discrepancy,
            1e-4,
            f"Discrepancy {max_discrepancy:.3e} between Closed-Form and Gradient Descent exceeds 1e-4",
        )

    def test_lasso_sparsity_recovery(self) -> None:
        """Verify Coordinate Descent for Lasso correctly zeros out irrelevant features."""
        np.random.seed(42)
        # 100 samples, 10 features, only first 2 are active
        X = np.random.randn(100, 10)
        true_theta = np.zeros(10)
        true_theta[0] = 3.0
        true_theta[1] = -2.5
        y = X @ true_theta + np.random.randn(100) * 0.05

        lasso = MultivariatePolynomialRegression(
            degree=1,
            alpha=0.2,
            l1_ratio=1.0,
            solver="coordinate_descent",
            fit_intercept=False,
            max_iter=1000,
        )
        lasso.fit(X, y)

        coefs = lasso.coef_
        # Features 2 through 9 should be exactly zeroed out by soft-thresholding
        zero_coefs = coefs[2:]
        np.testing.assert_allclose(zero_coefs, 0.0, atol=1e-5)
        self.assertGreater(abs(coefs[0]), 2.0)
        self.assertGreater(abs(coefs[1]), 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

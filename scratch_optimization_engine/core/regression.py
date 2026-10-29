"""
Multivariate Polynomial Regression with Elastic Net, Lasso (L1), and Ridge (L2) Regularization
==============================================================================================

This module provides an industrial-grade, dependency-free implementation of
Multivariate Polynomial Feature mapping and Linear Regression solvers:
- Closed-Form Solution: Normal Equation via direct Cholesky and QR matrix factorizations.
- Iterative Vectorized Gradient Descent: Supports custom first-order optimizers (SGD, RMSProp, Adam)
  with Proximal Gradient Descent (Iterative Shrinkage-Thresholding Algorithm - ISTA).
- Cyclical Coordinate Descent: Exact coordinate-wise minimization for L1/L2 regularized objectives.

All operations are implemented exclusively using NumPy vectorization without
scikit-learn or scipy.optimize.

Mathematical Formulation:
-------------------------
Given feature matrix $X \\in \\mathbb{R}^{n \\times d}$ and targets $y \\in \\mathbb{R}^n$,
we construct polynomial feature tensor $\\Phi(X) \\in \\mathbb{R}^{n \\times D}$ of degree $p$,
where $D = \\binom{d+p}{p}$.

The regularized objective function is:
$$J(\\theta) = \\frac{1}{2n} \\|\\Phi(X)\\theta - y\\|_2^2 + \\alpha \\left[ \\rho \\|\\theta_{1:}\\|_1 + \\frac{1-\\rho}{2} \\|\\theta_{1:}\\|_2^2 \\right]$$
where $\\theta_0$ represents the unpenalized intercept parameter.

Normal Equation (Ridge, $\\rho = 0$):
$$\\theta = (\\Phi^T \\Phi + n \\alpha I')^{-1} \\Phi^T y$$
where $I' = \\text{diag}(0, 1, 1, \\dots, 1) \\in \\mathbb{R}^{D \\times D}$.
"""

from __future__ import annotations
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scratch_optimization_engine.core.optimizers import Optimizer, get_optimizer
except (ModuleNotFoundError, ImportError):
    from core.optimizers import Optimizer, get_optimizer


def soft_thresholding(z: np.ndarray, threshold: float) -> np.ndarray:
    """
    Vectorized Soft-Thresholding (Shrinkage) Operator $\\mathcal{S}_{\\lambda}(z)$.

    Mathematical Definition:
    ------------------------
    $$\\mathcal{S}_{\\lambda}(z) = \\text{sign}(z) \\cdot \\max(|z| - \\lambda, 0)$$

    Parameters
    ----------
    z : np.ndarray
        Input coordinate values.
    threshold : float
        Shrinkage penalty parameter $\\lambda \\ge 0$.

    Returns
    -------
    np.ndarray
        Thresholded coordinate array with identical shape as `z`.

    Complexity:
    -----------
    - Time: $O(d)$ where $d = \\text{len}(z)$.
    - Space: $O(d)$ output array.
    """
    if threshold <= 0.0:
        return z
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve symmetric positive-definite linear system $A x = b$ using Cholesky Decomposition.

    Mathematical Algorithm:
    -----------------------
    1. Factorize $A = L L^T$ where $L$ is lower triangular with positive diagonal.
    2. Forward substitution: solve $L z = b$.
    3. Backward substitution: solve $L^T x = z$.

    Parameters
    ----------
    A : np.ndarray
        Symmetric positive-definite coefficient matrix $\\in \\mathbb{R}^{D \\times D}$.
    b : np.ndarray
        Right-hand side vector $\\in \\mathbb{R}^D$ or matrix $\\in \\mathbb{R}^{D \\times k}$.

    Returns
    -------
    np.ndarray
        Solution array $x \\in \\mathbb{R}^D$.

    Complexity:
    -----------
    - Time: $\\frac{1}{3} D^3 + 2 D^2$ floating point operations.
    - Space: $O(D^2)$ matrix storage.
    """
    try:
        L = np.linalg.cholesky(A)
        # Forward substitution: L z = b
        z = np.linalg.solve(L, b)
        # Backward substitution: L^T x = z
        x = np.linalg.solve(L.T, z)
        return x
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse via SVD if matrix is ill-conditioned or singular
        return np.linalg.pinv(A) @ b


class PolynomialFeatures:
    """
    Multivariate Polynomial Feature Expansion.

    Generates all monomial combinations of feature variables $\\prod_{j=1}^d x_j^{k_j}$
    such that degree $\\sum_{j=1}^d k_j \\le p$.

    Parameters
    ----------
    degree : int
        Maximal polynomial degree $p \\ge 1$ (default: 2).
    include_bias : bool
        If True, prepend a column of ones representing intercept (default: True).
    interaction_only : bool
        If True, only produce interaction terms without powers $x_j^k$ for $k > 1$ (default: False).

    Attributes
    ----------
    powers_ : np.ndarray
        Exponent matrix of shape $(D, d)$, detailing powers for each output column.
    n_input_features_ : int
        Dimensionality $d$ of input data.
    n_output_features_ : int
        Total dimensionality $D$ of expanded feature space.

    Complexity:
    -----------
    - Output Feature Dimension: $D = \\binom{d + p}{p}$.
    - Transform Time: $O(n \\cdot D)$ where $n$ is sample size.
    - Transform Space: $O(n \\cdot D)$ for the expanded matrix.
    """

    def __init__(self, degree: int = 2, include_bias: bool = True, interaction_only: bool = False) -> None:
        if degree < 1:
            raise ValueError(f"Degree must be an integer >= 1, got {degree}")
        self.degree: int = int(degree)
        self.include_bias: bool = bool(include_bias)
        self.interaction_only: bool = bool(interaction_only)

        self.powers_: Optional[np.ndarray] = None
        self.n_input_features_: int = 0
        self.n_output_features_: int = 0

    def fit(self, X: np.ndarray) -> "PolynomialFeatures":
        """
        Compute feature combination index mappings and power exponents.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape $(n, d)$.

        Returns
        -------
        PolynomialFeatures
            Self reference.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_input_features_ = n_features

        combinations: List[Tuple[int, ...]] = []
        if self.include_bias:
            combinations.append(())

        comb_func = itertools.combinations if self.interaction_only else itertools.combinations_with_replacement

        for d in range(1, self.degree + 1):
            for combo in comb_func(range(n_features), d):
                combinations.append(combo)

        self.n_output_features_ = len(combinations)
        powers = np.zeros((self.n_output_features_, n_features), dtype=np.int64)

        for col_idx, combo in enumerate(combinations):
            for feature_idx in combo:
                powers[col_idx, feature_idx] += 1

        self.powers_ = powers
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform raw input features $X \\in \\mathbb{R}^{n \\times d}$ into polynomial space $\\Phi(X) \\in \\mathbb{R}^{n \\times D}$.

        Parameters
        ----------
        X : np.ndarray
            Input dataset of shape $(n, d)$.

        Returns
        -------
        np.ndarray
            Polynomial feature tensor of shape $(n, D)$.
        """
        if self.powers_ is None:
            self.fit(X)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        if n_features != self.n_input_features_:
            raise ValueError(f"Input features {n_features} does not match fitted features {self.n_input_features_}")

        # Vectorized polynomial evaluation
        D = self.n_output_features_
        output = np.empty((n_samples, D), dtype=np.float64)

        for col_idx in range(D):
            power_vec = self.powers_[col_idx]
            if np.all(power_vec == 0):
                output[:, col_idx] = 1.0
            else:
                # Active feature indices and corresponding exponents
                active_indices = np.nonzero(power_vec)[0]
                col_vals = np.ones(n_samples, dtype=np.float64)
                for idx in active_indices:
                    exponent = power_vec[idx]
                    if exponent == 1:
                        col_vals *= X[:, idx]
                    else:
                        col_vals *= (X[:, idx] ** exponent)
                output[:, col_idx] = col_vals

        return output

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in a single pass."""
        return self.fit(X).transform(X)

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Generate human-readable names for the polynomial features."""
        if self.powers_ is None:
            raise ValueError("PolynomialFeatures instance is not fitted yet.")

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]

        names: List[str] = []
        for power_vec in self.powers_:
            if np.all(power_vec == 0):
                names.append("1")
                continue
            terms = []
            for idx, p in enumerate(power_vec):
                if p == 1:
                    terms.append(input_features[idx])
                elif p > 1:
                    terms.append(f"{input_features[idx]}^{p}")
            names.append(" * ".join(terms))
        return names


class MultivariatePolynomialRegression:
    """
    Multivariate Polynomial Regression with Elastic Net, Lasso (L1), and Ridge (L2) Regularization.

    Supports:
    - `solver='closed_form'`: Solves Normal Equation analytically via Cholesky decomposition.
    - `solver='gradient_descent'`: Optimizes parameters via first-order optimizers (SGD, RMSProp, Adam)
      combined with Proximal Gradient Descent (ISTA) for non-smooth L1 penalties.
    - `solver='coordinate_descent'`: Direct cyclical coordinate descent minimization.

    Loss Formulation:
    -----------------
    $$J(\\theta) = \\frac{1}{2n} \\|X\\theta - y\\|_2^2 + \\alpha \\left[ \\rho \\|\\theta_{1:}\\|_1 + \\frac{1-\\rho}{2} \\|\\theta_{1:}\\|_2^2 \\right]$$
    where $\\theta_0$ is the unpenalized intercept parameter.

    Parameters
    ----------
    degree : int
        Degree $p$ of polynomial feature expansion (default: 1 for standard linear regression).
    alpha : float
        Overall regularization weight $\\alpha \\ge 0$ (default: 0.0).
    l1_ratio : float
        ElasticNet mixing parameter $\\rho \\in [0, 1]$ (default: 0.0 for pure Ridge L2, 1.0 for pure Lasso L1).
    fit_intercept : bool
        Whether to calculate an unpenalized bias term (default: True).
    solver : str
        Optimization strategy: 'closed_form', 'gradient_descent', or 'coordinate_descent' (default: 'closed_form').
    optimizer : Union[Optimizer, str]
        First-order optimizer instance or string identifier for 'gradient_descent' (default: 'adam').
    learning_rate : float
        Learning rate $\\eta$ for gradient descent (default: 0.01).
    max_iter : int
        Maximum number of iterations / epochs (default: 2000).
    tol : float
        Convergence threshold on parameter change or loss decrease (default: 1e-7).
    verbose : bool
        Whether to log optimization diagnostics (default: False).

    Complexity:
    -----------
    - Closed-Form: Time $O(n D^2 + D^3)$, Space $O(D^2 + n D)$ where $D = \\binom{d+p}{p}$.
    - Gradient Descent: Time $O(K \\cdot n D)$ where $K = \\text{iterations}$, Space $O(n D + D)$.
    - Coordinate Descent: Time $O(K \\cdot n D)$, Space $O(n D + D)$.
    """

    def __init__(
        self,
        degree: int = 1,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        solver: str = "closed_form",
        optimizer: Union[Optimizer, str] = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 2000,
        tol: float = 1e-7,
        verbose: bool = False,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"Alpha regularization parameter must be >= 0, got {alpha}")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")

        valid_solvers = ("closed_form", "gradient_descent", "coordinate_descent")
        if solver not in valid_solvers:
            raise ValueError(f"Invalid solver '{solver}'. Expected one of {valid_solvers}")

        self.degree: int = int(degree)
        self.alpha: float = float(alpha)
        self.l1_ratio: float = float(l1_ratio)
        self.fit_intercept: bool = bool(fit_intercept)
        self.solver: str = solver
        self.learning_rate: float = float(learning_rate)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.verbose: bool = bool(verbose)

        if isinstance(optimizer, str):
            self.optimizer: Optimizer = get_optimizer(optimizer, learning_rate=self.learning_rate)
        else:
            self.optimizer = optimizer

        self.poly_: Optional[PolynomialFeatures] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.theta_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    def _prepare_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply polynomial expansion and intercept handling."""
        if fit:
            self.poly_ = PolynomialFeatures(
                degree=self.degree,
                include_bias=self.fit_intercept,
                interaction_only=False,
            )
            return self.poly_.fit_transform(X)
        else:
            if self.poly_ is None:
                raise ValueError("Model has not been fitted yet.")
            return self.poly_.transform(X)

    def compute_loss(self, Phi: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute total regularized objective value $J(\\theta)$.

        $$J(\\theta) = \\frac{1}{2n} \\|\\Phi\\theta - y\\|_2^2 + \\alpha \\left[ \\rho \\|\\theta_{1:}\\|_1 + \\frac{1-\\rho}{2} \\|\\theta_{1:}\\|_2^2 \\right]$$

        Parameters
        ----------
        Phi : np.ndarray
            Expanded feature matrix of shape $(n, D)$.
        y : np.ndarray
            Target vector of shape $(n,)$.
        theta : np.ndarray
            Parameter vector of shape $(D,)$.

        Returns
        -------
        float
            Objective loss value.
        """
        n_samples = Phi.shape[0]
        residuals = Phi @ theta - y
        mse_loss = 0.5 * np.mean(residuals ** 2)

        if self.alpha == 0.0:
            return float(mse_loss)

        # Intercept is at index 0 if fit_intercept is True
        reg_params = theta[1:] if self.fit_intercept else theta
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(reg_params))
        l2_penalty = 0.5 * self.alpha * (1.0 - self.l1_ratio) * np.sum(reg_params ** 2)

        return float(mse_loss + l1_penalty + l2_penalty)

    def compute_gradient_smooth(self, Phi: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute analytical gradient of the smooth loss component (MSE + L2 Ridge).

        $$\\nabla_{\\theta} J_{\\text{smooth}}(\\theta) = \\frac{1}{n} \\Phi^T (\\Phi\\theta - y) + \\alpha(1 - \\rho) \\tilde{\\theta}$$
        where $\\tilde{\\theta} = [0, \\theta_1, \\dots, \\theta_{D-1}]^T$.

        Parameters
        ----------
        Phi : np.ndarray
            Feature matrix $(n, D)$.
        y : np.ndarray
            Target vector $(n,)$.
        theta : np.ndarray
            Parameter vector $(D,)$.

        Returns
        -------
        np.ndarray
            Gradient vector $\\in \\mathbb{R}^D$.
        """
        n_samples = Phi.shape[0]
        residuals = Phi @ theta - y
        grad = (Phi.T @ residuals) / n_samples

        if self.alpha > 0.0 and self.l1_ratio < 1.0:
            reg_theta = theta.copy()
            if self.fit_intercept:
                reg_theta[0] = 0.0
            grad += self.alpha * (1.0 - self.l1_ratio) * reg_theta

        return grad

    def _fit_closed_form(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Analytical solution via Normal Equation:
        $$\\theta = (\\Phi^T \\Phi + n \\alpha (1 - \\rho) I')^{-1} \\Phi^T y$$
        """
        if self.alpha > 0.0 and self.l1_ratio > 0.0:
            raise ValueError(
                "Closed-form solver cannot solve non-smooth L1 (Lasso) penalty. "
                "Use solver='gradient_descent' or solver='coordinate_descent'."
            )

        n_samples, D = Phi.shape
        # Compute Gram matrix Phi^T Phi and right-hand side Phi^T y
        A = (Phi.T @ Phi) / n_samples
        b = (Phi.T @ y) / n_samples

        if self.alpha > 0.0:
            reg_matrix = np.eye(D, dtype=np.float64) * (self.alpha * (1.0 - self.l1_ratio))
            if self.fit_intercept:
                reg_matrix[0, 0] = 0.0  # Do not regularize intercept
            A += reg_matrix

        theta = cholesky_solve(A, b)
        return theta

    def _fit_gradient_descent(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Iterative optimization via first-order optimizer with Proximal Gradient step (ISTA).
        """
        n_samples, D = Phi.shape
        theta = np.zeros(D, dtype=np.float64)
        self.optimizer.reset()
        self.loss_history_ = []

        eta = self.learning_rate
        l1_lambda = self.alpha * self.l1_ratio

        for epoch in range(self.max_iter):
            current_loss = self.compute_loss(Phi, y, theta)
            self.loss_history_.append(current_loss)

            # Gradient of smooth portion (MSE + L2)
            grad_smooth = self.compute_gradient_smooth(Phi, y, theta)

            # First-order optimizer step
            theta_next = self.optimizer.step(theta, grad_smooth)

            # Proximal step (Soft-Thresholding) for L1 penalty
            if l1_lambda > 0.0:
                threshold = eta * l1_lambda
                if self.fit_intercept:
                    intercept_val = theta_next[0]
                    theta_next = soft_thresholding(theta_next, threshold)
                    theta_next[0] = intercept_val
                else:
                    theta_next = soft_thresholding(theta_next, threshold)

            # Check parameter convergence
            param_delta = np.linalg.norm(theta_next - theta) / (np.linalg.norm(theta) + 1e-12)
            theta = theta_next

            if epoch > 10 and param_delta < self.tol:
                if self.verbose:
                    print(f"Gradient Descent converged at epoch {epoch} (delta: {param_delta:.2e})")
                break

        self.loss_history_.append(self.compute_loss(Phi, y, theta))
        return theta

    def _fit_coordinate_descent(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Cyclical Coordinate Descent for ElasticNet / Lasso / Ridge.
        """
        n_samples, D = Phi.shape
        theta = np.zeros(D, dtype=np.float64)
        self.loss_history_ = []

        # Precompute column squared norms: z_j = ||Phi_j||^2 / n
        col_norms = np.sum(Phi ** 2, axis=0) / n_samples
        residuals = y - Phi @ theta

        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(self.max_iter):
            current_loss = self.compute_loss(Phi, y, theta)
            self.loss_history_.append(current_loss)
            max_coord_change = 0.0

            for j in range(D):
                old_theta_j = theta[j]

                # Remove contribution of coordinate j from residuals
                if old_theta_j != 0.0:
                    residuals += Phi[:, j] * old_theta_j

                # Correlation rho_j = (Phi_j^T residuals) / n
                rho_j = float(np.dot(Phi[:, j], residuals) / n_samples)

                if self.fit_intercept and j == 0:
                    # Unpenalized intercept coordinate
                    new_theta_j = rho_j / max(col_norms[j], 1e-12)
                else:
                    # ElasticNet thresholding
                    shrinkage = soft_thresholding(np.array([rho_j]), l1_penalty)[0]
                    denominator = col_norms[j] + l2_penalty
                    new_theta_j = shrinkage / max(denominator, 1e-12)

                theta[j] = new_theta_j
                # Re-apply updated contribution to residuals
                if new_theta_j != 0.0:
                    residuals -= Phi[:, j] * new_theta_j

                change = abs(new_theta_j - old_theta_j)
                if change > max_coord_change:
                    max_coord_change = change

            if max_coord_change < self.tol:
                if self.verbose:
                    print(f"Coordinate Descent converged at iteration {iteration} (max change: {max_coord_change:.2e})")
                break

        self.loss_history_.append(self.compute_loss(Phi, y, theta))
        return theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultivariatePolynomialRegression":
        """
        Fit regularized polynomial regression model to training dataset $(X, y)$.

        Parameters
        ----------
        X : np.ndarray
            Design matrix $\\in \\mathbb{R}^{n \\times d}$.
        y : np.ndarray
            Target labels $\\in \\mathbb{R}^n$.

        Returns
        -------
        MultivariatePolynomialRegression
            Fitted estimator instance.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Phi = self._prepare_features(X, fit=True)

        if self.solver == "closed_form":
            self.theta_ = self._fit_closed_form(Phi, y)
        elif self.solver == "gradient_descent":
            self.theta_ = self._fit_gradient_descent(Phi, y)
        elif self.solver == "coordinate_descent":
            self.theta_ = self._fit_coordinate_descent(Phi, y)
        else:
            raise ValueError(f"Unknown solver {self.solver}")

        if self.fit_intercept:
            self.intercept_ = float(self.theta_[0])
            self.coef_ = self.theta_[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta_.copy()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous target values for feature matrix $X$.

        Parameters
        ----------
        X : np.ndarray
            Input features $\\in \\mathbb{R}^{m \\times d}$.

        Returns
        -------
        np.ndarray
            Predictions $\\hat{y} \\in \\mathbb{R}^m$.
        """
        if self.theta_ is None:
            raise ValueError("Model is not fitted yet.")

        Phi = self._prepare_features(X, fit=False)
        return Phi @ self.theta_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute coefficient of determination $R^2$ score:
        $$R^2 = 1 - \\frac{\\sum_{i=1}^n (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^n (y_i - \\bar{y})^2}$$

        Parameters
        ----------
        X : np.ndarray
            Test feature matrix.
        y : np.ndarray
            True target values.

        Returns
        -------
        float
            $R^2$ determination score $\\le 1.0$.
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        if v == 0.0:
            return 0.0
        return float(1.0 - (u / v))

"""
Numerical Optimization Engine - First-Order Stochastic & Deterministic Optimizers
=================================================================================

This module implements first-order optimization algorithms from scratch using
vectorized NumPy operations without high-level external frameworks.

Implemented Algorithms:
-----------------------
1. SGD (Stochastic Gradient Descent) with Polyak Momentum & Nesterov Accelerated Gradient.
2. RMSProp (Root Mean Square Propagation) with adaptive learning rates.
3. Adam (Adaptive Moment Estimation) with exact bias corrections and AMSGrad support.

Mathematical Complexity:
------------------------
For parameter vector $\\theta \\in \\mathbb{R}^d$:
- Time Complexity: $O(d)$ per parameter update step across all algorithms.
- Space Complexity:
    - SGD (no momentum): $O(1)$ auxiliary memory.
    - SGD (with momentum): $O(d)$ auxiliary state for velocity vector $v_t$.
    - RMSProp: $O(d)$ auxiliary state for squared gradient moving average $v_t$.
    - Adam: $2 \\cdot O(d) = O(d)$ auxiliary state for first and second moment vectors $(m_t, v_t)$.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np


class Optimizer(ABC):
    """
    Abstract Base Class for First-Order Parameter Optimizers.

    Defines the contract for updating parameter arrays $\\theta \\in \\mathbb{R}^d$
    given analytical gradients $g_t = \\nabla_{\\theta} J(\\theta_t)$.

    Parameters
    ----------
    learning_rate : float
        Base step size $\\eta > 0$.
    weight_decay : float, optional
        L2 regularization coefficient $\\lambda \\ge 0$ added directly to gradients
        (default is 0.0).
    """

    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0) -> None:
        if learning_rate <= 0.0:
            raise ValueError(f"Learning rate must be strictly positive, got {learning_rate}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")

        self.lr: float = float(learning_rate)
        self.weight_decay: float = float(weight_decay)
        self.step_count: int = 0
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Execute a single parameter update step.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector $\\theta_t \\in \\mathbb{R}^d$ or matrix.
        grads : np.ndarray
            Evaluated analytical gradient $\\nabla_{\\theta} J(\\theta_t) \\in \\mathbb{R}^d$,
            matching the shape of `params`.

        Returns
        -------
        np.ndarray
            Updated parameter array $\\theta_{t+1}$, same shape as `params`.
        """
        pass

    def reset(self) -> None:
        """Reset internal optimization state and step counter to zero."""
        self.step_count = 0
        self.state.clear()

    def state_dict(self) -> Dict[str, Any]:
        """
        Export optimizer state dictionary for checkpointing and inspection.

        Returns
        -------
        Dict[str, Any]
            State dictionary containing iteration count, hyperparameters, and moment arrays.
        """
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "state": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.state.items()}
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore optimizer state from a serialized state dictionary.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            State dictionary previously returned by `state_dict()`.
        """
        self.step_count = int(state_dict["step_count"])
        self.lr = float(state_dict["lr"])
        self.weight_decay = float(state_dict["weight_decay"])
        self.state = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state_dict["state"].items()}


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with Classical Momentum & Nesterov Acceleration.

    Mathematical Formulation:
    -------------------------
    Given gradient $g_t = \\nabla J(\\theta_{t-1}) + \\lambda \\theta_{t-1}$:

    Standard Momentum (Polyak, 1964):
    $$v_t = \\mu v_{t-1} + (1 - \\tau) g_t$$
    $$\\theta_t = \\theta_{t-1} - \\eta v_t$$

    Nesterov Accelerated Gradient (NAG, Nesterov 1983):
    $$v_t = \\mu v_{t-1} + g_t$$
    $$\\theta_t = \\theta_{t-1} - \\eta (g_t + \\mu v_t)$$

    Parameters
    ----------
    learning_rate : float
        Step size $\\eta > 0$ (default: 0.01).
    momentum : float
        Momentum damping coefficient $\\mu \\in [0, 1)$ (default: 0.0).
    dampening : float
        Dampening for momentum $\\tau \\in [0, 1]$ (default: 0.0).
    weight_decay : float
        L2 weight decay factor $\\lambda \\ge 0$ (default: 0.0).
    nesterov : bool
        Whether to enable Nesterov Accelerated Gradient (default: False).

    Complexity:
    -----------
    - Time: $O(d)$ floating-point operations per step.
    - Space: $O(d)$ auxiliary memory for velocity buffer $v_t$.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum factor must be in [0, 1), got {momentum}")
        if not 0.0 <= dampening <= 1.0:
            raise ValueError(f"Dampening factor must be in [0, 1], got {dampening}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")

        self.momentum: float = float(momentum)
        self.dampening: float = float(dampening)
        self.nesterov: bool = bool(nesterov)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Execute SGD update on parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter tensor $\\theta_{t-1}$.
        grads : np.ndarray
            Gradient tensor $g_t$.

        Returns
        -------
        np.ndarray
            Updated parameter tensor $\\theta_t$.
        """
        self.step_count += 1
        g = np.array(grads, dtype=np.float64, copy=True)

        if self.weight_decay != 0.0:
            g += self.weight_decay * params

        if self.momentum != 0.0:
            if "velocity" not in self.state:
                self.state["velocity"] = np.zeros_like(params, dtype=np.float64)

            v = self.state["velocity"]
            v = self.momentum * v + (1.0 - self.dampening) * g
            self.state["velocity"] = v

            if self.nesterov:
                update_direction = g + self.momentum * v
            else:
                update_direction = v
        else:
            update_direction = g

        new_params = params - self.lr * update_direction
        return new_params


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation (RMSProp) Adaptive Learning Rate Optimizer.

    Maintains an exponentially decaying average of squared gradients to scale
    coordinate updates inversely by gradient magnitude.

    Mathematical Formulation:
    -------------------------
    $$v_t = \\alpha v_{t-1} + (1 - \\alpha) g_t^2$$
    $$\\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} g_t$$

    With momentum $\\mu > 0$:
    $$m_t = \\mu m_{t-1} + \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} g_t$$
    $$\\theta_t = \\theta_{t-1} - m_t$$

    Parameters
    ----------
    learning_rate : float
        Step size $\\eta > 0$ (default: 0.001).
    alpha : float
        Smoothing constant $\\alpha \\in (0, 1)$ for second moment EMA (default: 0.99).
    eps : float
        Small positive constant $\\epsilon > 0$ preventing division by zero (default: 1e-8).
    weight_decay : float
        L2 weight decay factor $\\lambda \\ge 0$ (default: 0.0).
    momentum : float
        Momentum factor $\\mu \\in [0, 1)$ (default: 0.0).

    Complexity:
    -----------
    - Time: $O(d)$ operations per step.
    - Space: $O(d)$ for second moment buffer $v_t$ (plus $O(d)$ if momentum > 0).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha smoothing constant must be in (0, 1), got {alpha}")
        if eps <= 0.0:
            raise ValueError(f"Epsilon must be strictly positive, got {eps}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum factor must be in [0, 1), got {momentum}")

        self.alpha: float = float(alpha)
        self.eps: float = float(eps)
        self.momentum: float = float(momentum)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Execute RMSProp update on parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter tensor $\\theta_{t-1}$.
        grads : np.ndarray
            Gradient tensor $g_t$.

        Returns
        -------
        np.ndarray
            Updated parameter tensor $\\theta_t$.
        """
        self.step_count += 1
        g = np.array(grads, dtype=np.float64, copy=True)

        if self.weight_decay != 0.0:
            g += self.weight_decay * params

        if "square_avg" not in self.state:
            self.state["square_avg"] = np.zeros_like(params, dtype=np.float64)

        v = self.state["square_avg"]
        v = self.alpha * v + (1.0 - self.alpha) * (g ** 2)
        self.state["square_avg"] = v

        scaled_g = g / (np.sqrt(v) + self.eps)

        if self.momentum > 0.0:
            if "momentum_buffer" not in self.state:
                self.state["momentum_buffer"] = np.zeros_like(params, dtype=np.float64)
            m = self.state["momentum_buffer"]
            m = self.momentum * m + scaled_g
            self.state["momentum_buffer"] = m
            update_delta = self.lr * m
        else:
            update_delta = self.lr * scaled_g

        return params - update_delta


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) Optimizer.

    Maintains running exponential moving averages of both first-order moments
    (mean gradient $m_t$) and second-order raw moments (uncentered variance $v_t$),
    with analytically derived bias correction factors for initialization transients.

    Mathematical Formulation:
    -------------------------
    $$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$
    $$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$

    Analytical Bias Correction:
    $$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$$
    $$\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$

    AMSGrad Variant (Reddi et al., 2018):
    $$\\hat{v}_{t,\\max} = \\max(\\hat{v}_{t-1,\\max}, \\hat{v}_t)$$
    $$\\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_{t,\\max}} + \\epsilon} \\hat{m}_t$$

    Standard Parameter Update:
    $$\\theta_t = \\theta_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$

    Parameters
    ----------
    learning_rate : float
        Step size $\\eta > 0$ (default: 0.001).
    betas : Tuple[float, float]
        Coefficients $(\\beta_1, \\beta_2) \\in [0, 1) \\times [0, 1)$ for computing
        running averages of gradient and its square (default: (0.9, 0.999)).
    eps : float
        Small positive constant $\\epsilon > 0$ for numerical stabilization (default: 1e-8).
    weight_decay : float
        L2 weight decay coefficient $\\lambda \\ge 0$ (default: 0.0).
    amsgrad : bool
        Whether to enforce monotonically non-decreasing second moment bounds (default: False).

    Complexity:
    -----------
    - Time: $O(d)$ operations per parameter update step.
    - Space: $2 \\cdot O(d)$ for moment vectors $m_t$ and $v_t$ (plus $O(d)$ if amsgrad=True).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Epsilon must be strictly positive, got {eps}")

        self.beta1: float = float(beta1)
        self.beta2: float = float(beta2)
        self.eps: float = float(eps)
        self.amsgrad: bool = bool(amsgrad)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Execute Adam update on parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter tensor $\\theta_{t-1}$.
        grads : np.ndarray
            Gradient tensor $g_t$.

        Returns
        -------
        np.ndarray
            Updated parameter tensor $\\theta_t$.
        """
        self.step_count += 1
        t = self.step_count
        g = np.array(grads, dtype=np.float64, copy=True)

        if self.weight_decay != 0.0:
            g += self.weight_decay * params

        if "m" not in self.state:
            self.state["m"] = np.zeros_like(params, dtype=np.float64)
            self.state["v"] = np.zeros_like(params, dtype=np.float64)
            if self.amsgrad:
                self.state["v_max"] = np.zeros_like(params, dtype=np.float64)

        m = self.state["m"]
        v = self.state["v"]

        # Exponential moving averages of first and second moments
        m = self.beta1 * m + (1.0 - self.beta1) * g
        v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

        self.state["m"] = m
        self.state["v"] = v

        # Bias corrections
        bias_correction1 = 1.0 - (self.beta1 ** t)
        bias_correction2 = 1.0 - (self.beta2 ** t)

        m_hat = m / bias_correction1
        v_hat = v / bias_correction2

        if self.amsgrad:
            v_max = self.state["v_max"]
            v_max = np.maximum(v_max, v_hat)
            self.state["v_max"] = v_max
            denom = np.sqrt(v_max) + self.eps
        else:
            denom = np.sqrt(v_hat) + self.eps

        new_params = params - self.lr * (m_hat / denom)
        return new_params


def get_optimizer(name: str, **kwargs: Any) -> Optimizer:
    """
    Factory method to instantiate optimizers by identifier name.

    Parameters
    ----------
    name : str
        Optimizer identifier: 'sgd', 'momentum', 'rmsprop', or 'adam'.
    **kwargs : Any
        Hyperparameters passed to the optimizer constructor.

    Returns
    -------
    Optimizer
        Instantiated optimizer instance.
    """
    canonical = name.strip().lower()
    if canonical in ("sgd", "vanilla_sgd"):
        return SGD(**kwargs)
    elif canonical in ("momentum", "sgd_momentum"):
        kwargs.setdefault("momentum", 0.9)
        return SGD(**kwargs)
    elif canonical in ("rmsprop", "rms"):
        return RMSProp(**kwargs)
    elif canonical in ("adam", "amsgrad"):
        return Adam(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Options: 'sgd', 'momentum', 'rmsprop', 'adam'")

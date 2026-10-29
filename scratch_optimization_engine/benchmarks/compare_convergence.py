"""
Empirical Convergence, Wall-Clock Execution Time, and Memory Profiling Benchmark
==============================================================================

Conducts rigorous empirical comparisons of first-order optimizers:
1. Vanilla Stochastic Gradient Descent (SGD)
2. SGD with Polyak Momentum (mu = 0.9)
3. Root Mean Square Propagation (RMSProp)
4. Adaptive Moment Estimation (Adam)

Benchmarked Problem Formulations:
---------------------------------
1. High-Dimensional Multivariate Polynomial Regression:
   Ill-conditioned feature matrix with severe multicollinearity.
2. Non-Convex Rosenbrock Valley Function:
   $$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$
   Global minimum at $(1, 1)$ with $f(1, 1) = 0$.

Metrics Profiled:
-----------------
- Loss Trajectory Curves across iterations (log scale).
- Wall-Clock Run Time measured via `time.perf_counter()`.
- Peak Heap Memory Usage tracked via `tracemalloc`.
- Multi-panel visual diagnostic figure saved to `benchmarks/convergence_comparison.png`.
"""

from __future__ import annotations
import os
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Tuple

# Configure Matplotlib backend and cache
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure both parent workspace and scratch_optimization_engine directory are on sys.path
_bench_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_bench_dir)
_parent_dir = os.path.dirname(_engine_dir)
for _p in [_parent_dir, _engine_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scratch_optimization_engine.core.optimizers import SGD, RMSProp, Adam, Optimizer
    from scratch_optimization_engine.core.regression import MultivariatePolynomialRegression, PolynomialFeatures
except (ModuleNotFoundError, ImportError):
    from core.optimizers import SGD, RMSProp, Adam, Optimizer
    from core.regression import MultivariatePolynomialRegression, PolynomialFeatures


def rosenbrock_function(xy: np.ndarray) -> float:
    """
    Evaluate 2D Rosenbrock function:
    $$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$
    """
    x, y = xy[0], xy[1]
    return float((1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2)


def rosenbrock_gradient(xy: np.ndarray) -> np.ndarray:
    """
    Analytical gradient of 2D Rosenbrock function:
    $$\\nabla f(x, y) = \\begin{bmatrix} -2(1 - x) - 400 x (y - x^2) \\\\ 200(y - x^2) \\end{bmatrix}$$
    """
    x, y = xy[0], xy[1]
    df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x ** 2)
    df_dy = 200.0 * (y - x ** 2)
    return np.array([df_dx, df_dy], dtype=np.float64)


def profile_rosenbrock_optimizer(
    optimizer_cls: type[Optimizer],
    opt_kwargs: Dict[str, Any],
    start_pos: np.ndarray = np.array([-1.5, 2.0]),
    iterations: int = 1500,
) -> Dict[str, Any]:
    """Profile optimizer trajectory, runtime, and memory on Rosenbrock function."""
    tracemalloc.start()
    t_start = time.perf_counter()

    optimizer = optimizer_cls(**opt_kwargs)
    pos = start_pos.copy().astype(np.float64)

    trajectory: List[np.ndarray] = [pos.copy()]
    losses: List[float] = [rosenbrock_function(pos)]

    converged_iter = iterations
    for it in range(1, iterations + 1):
        grad = rosenbrock_gradient(pos)
        pos = optimizer.step(pos, grad)
        trajectory.append(pos.copy())
        current_loss = rosenbrock_function(pos)
        losses.append(current_loss)

        if current_loss < 1e-4 and converged_iter == iterations:
            converged_iter = it

    t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "trajectory": np.array(trajectory),
        "losses": np.array(losses),
        "final_loss": losses[-1],
        "final_pos": pos,
        "time_ms": t_elapsed,
        "peak_mem_kib": peak_mem / 1024.0,
        "converged_iter": converged_iter,
    }


def benchmark_regression_optimizers(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    max_iter: int = 1500,
) -> Dict[str, Dict[str, Any]]:
    """Benchmark all optimizers on Multivariate Polynomial Regression."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(X)
    D = Phi.shape[1]

    configs = {
        "SGD (Vanilla)": (SGD, {"learning_rate": 0.005, "momentum": 0.0}),
        "SGD (Momentum)": (SGD, {"learning_rate": 0.005, "momentum": 0.9}),
        "RMSProp": (RMSProp, {"learning_rate": 0.01, "alpha": 0.99}),
        "Adam": (Adam, {"learning_rate": 0.05, "betas": (0.9, 0.999)}),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, (opt_cls, kwargs) in configs.items():
        tracemalloc.start()
        t_start = time.perf_counter()

        opt = opt_cls(**kwargs)
        theta = np.zeros(D, dtype=np.float64)
        losses: List[float] = []

        for it in range(max_iter):
            # Compute MSE loss and gradient
            residuals = Phi @ theta - y
            mse = float(0.5 * np.mean(residuals ** 2))
            losses.append(mse)

            grad = (Phi.T @ residuals) / Phi.shape[0]
            theta = opt.step(theta, grad)

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[name] = {
            "losses": np.array(losses),
            "final_loss": losses[-1],
            "time_ms": t_elapsed,
            "peak_mem_kib": peak_mem / 1024.0,
            "theta": theta,
        }

    return results


def run_benchmarks() -> None:
    """Execute complete convergence and memory profiling benchmark suite."""
    print("=" * 80)
    print("  SCRATCH OPTIMIZATION ENGINE: FIRST-ORDER OPTIMIZER CONVERGENCE BENCHMARK  ")
    print("=" * 80)

    # 1. Synthesize Ill-Conditioned Multivariate Polynomial Dataset
    np.random.seed(42)
    n_samples, n_features = 120, 3
    # Introduce severe collinearity
    X_raw = np.random.randn(n_samples, n_features)
    X_raw[:, 2] = 0.9 * X_raw[:, 0] + 0.1 * X_raw[:, 1] + 0.05 * np.random.randn(n_samples)

    # True non-linear polynomial target
    y_true = (
        1.8
        + 2.5 * X_raw[:, 0]
        - 1.4 * X_raw[:, 1]
        + 0.8 * (X_raw[:, 0] ** 2)
        - 0.5 * (X_raw[:, 0] * X_raw[:, 1])
        + 0.2 * (X_raw[:, 2] ** 3)
        + np.random.randn(n_samples) * 0.1
    )

    print("\n[1/3] Benchmarking on Multivariate Polynomial Regression (Degree 3)...")
    reg_results = benchmark_regression_optimizers(X_raw, y_true, degree=3, max_iter=1500)

    print("\n--- Regression Optimization Performance Table ---")
    print(f"{'Optimizer':<18} | {'Final MSE Loss':<16} | {'Runtime (ms)':<14} | {'Peak Mem (KiB)':<14}")
    print("-" * 72)
    for name, res in reg_results.items():
        print(
            f"{name:<18} | {res['final_loss']:<16.6e} | {res['time_ms']:<14.2f} | {res['peak_mem_kib']:<14.2f}"
        )

    # 2. Benchmark on 2D Rosenbrock Banana Function
    print("\n[2/3] Benchmarking on Non-Convex Rosenbrock Valley Function...")
    start_point = np.array([-1.5, 2.0])
    rosen_configs = {
        "SGD (Vanilla)": (SGD, {"learning_rate": 0.001, "momentum": 0.0}),
        "SGD (Momentum)": (SGD, {"learning_rate": 0.001, "momentum": 0.9}),
        "RMSProp": (RMSProp, {"learning_rate": 0.01, "alpha": 0.99}),
        "Adam": (Adam, {"learning_rate": 0.05, "betas": (0.9, 0.999)}),
    }

    rosen_results: Dict[str, Dict[str, Any]] = {}
    for name, (opt_cls, kwargs) in rosen_configs.items():
        rosen_results[name] = profile_rosenbrock_optimizer(
            opt_cls, kwargs, start_pos=start_point, iterations=1500
        )

    print("\n--- Rosenbrock Optimization Performance Table ---")
    print(
        f"{'Optimizer':<18} | {'Final Loss':<14} | {'Convergence It':<16} | {'Runtime (ms)':<14} | {'Peak Mem (KiB)':<14}"
    )
    print("-" * 88)
    for name, res in rosen_results.items():
        conv_str = str(res['converged_iter']) if res['converged_iter'] < 1500 else "Did not reach 1e-4"
        print(
            f"{name:<18} | {res['final_loss']:<14.6e} | {conv_str:<16} | {res['time_ms']:<14.2f} | {res['peak_mem_kib']:<14.2f}"
        )

    # 3. Generate Visual Diagnostic Comparison Plots
    print("\n[3/3] Generating Convergence Diagnostic Figures...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {
        "SGD (Vanilla)": "#1f77b4",
        "SGD (Momentum)": "#ff7f0e",
        "RMSProp": "#2ca02c",
        "Adam": "#d62728",
    }

    # Plot 1: Polynomial Regression Loss Curves (log scale)
    ax1 = axes[0, 0]
    for name, res in reg_results.items():
        ax1.plot(res["losses"], label=name, color=colors[name], linewidth=2.0)
    ax1.set_yscale("log")
    ax1.set_xlabel("Optimization Epoch / Iteration", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Multivariate Polynomial Regression Convergence", fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Plot 2: Rosenbrock Loss Curves (log scale)
    ax2 = axes[0, 1]
    for name, res in rosen_results.items():
        ax2.plot(res["losses"], label=name, color=colors[name], linewidth=2.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("Optimization Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Rosenbrock Objective f(x, y) (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("Rosenbrock Banana Valley Convergence", fontsize=13, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    # Plot 3: 2D Parameter Trajectories on Rosenbrock Contour Landscape
    ax3 = axes[1, 0]
    x_grid = np.linspace(-2.0, 2.0, 400)
    y_grid = np.linspace(-1.0, 3.0, 400)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    Z_mesh = (1.0 - X_mesh) ** 2 + 100.0 * (Y_mesh - X_mesh ** 2) ** 2

    # Plot filled logarithmic contours
    contour_levels = np.logspace(-1, 3.5, 30)
    cs = ax3.contourf(X_mesh, Y_mesh, Z_mesh, levels=contour_levels, norm=matplotlib.colors.LogNorm(), cmap="viridis", alpha=0.6)
    fig.colorbar(cs, ax=ax3, label="f(x, y) (Log scale)")

    for name, res in rosen_results.items():
        traj = res["trajectory"]
        ax3.plot(traj[:, 0], traj[:, 1], label=name, color=colors[name], linewidth=2.2, alpha=0.9)
        ax3.scatter(traj[-1, 0], traj[-1, 1], color=colors[name], s=50, edgecolors="black", zorder=5)

    ax3.scatter(start_point[0], start_point[1], color="black", s=100, marker="x", label="Start (-1.5, 2.0)", zorder=6)
    ax3.scatter(1.0, 1.0, color="gold", s=140, marker="*", edgecolors="black", label="Optimum (1.0, 1.0)", zorder=6)
    ax3.set_xlabel("Parameter x", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Parameter y", fontsize=11, fontweight="bold")
    ax3.set_title("2D Parameter Space Trajectories", fontsize=13, fontweight="bold")
    ax3.set_xlim(-2.0, 2.0)
    ax3.set_ylim(-1.0, 3.0)
    ax3.legend(loc="lower left", frameon=True, fontsize=9)

    # Plot 4: Wall-Clock Runtime and Memory Comparison Bar Chart
    ax4 = axes[1, 1]
    names = list(reg_results.keys())
    x_pos = np.arange(len(names))
    bar_width = 0.35

    times = [reg_results[n]["time_ms"] for n in names]
    mems = [reg_results[n]["peak_mem_kib"] for n in names]

    ax4_twin = ax4.twinx()
    rects1 = ax4.bar(x_pos - bar_width / 2, times, bar_width, label="Runtime (ms)", color="#3498db", alpha=0.85)
    rects2 = ax4_twin.bar(x_pos + bar_width / 2, mems, bar_width, label="Peak Memory (KiB)", color="#e74c3c", alpha=0.85)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax4.set_ylabel("Wall-Clock Time (ms)", fontsize=11, fontweight="bold", color="#2980b9")
    ax4_twin.set_ylabel("Peak Memory (KiB)", fontsize=11, fontweight="bold", color="#c0392b")
    ax4.set_title("Runtime & Memory Profile (Regression)", fontsize=13, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "convergence_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Convergence benchmark plot successfully saved to:\n  {output_path}\n")


if __name__ == "__main__":
    run_benchmarks()

"""
2D Obstacle Grid State-Space Search and Smoothed Spline Trajectory Benchmark
===========================================================================

Demonstrates discrete A* state-space graph planning coupled with continuous
coordinate gradient descent trajectory smoothing and $C^1$-continuous Catmull-Rom
spline evaluation.

Benchmark Components:
---------------------
1. Synthetic 2D obstacle environment with mixed geometric primitives:
   - Rectangular walls forming narrow corridors and chokepoints.
   - Circular obstacles forcing curved evasive maneuvers.
2. Discrete A* grid search with Octile heuristic.
3. Gradient-based trajectory smoothing minimizing discrete biharmonic acceleration,
   elastic tension, and obstacle repulsive potential fields.
4. $C^1$ cubic Catmull-Rom spline trajectory generation with analytical curvature profile.
5. Quantitative path metric comparison:
   - Total Euclidean Path Length
   - Maximum & Integral Squared Curvature
   - Minimum & Mean Obstacle Clearance Distance
6. Multi-panel visual diagnostic figure saved to `benchmarks/trajectory_visualization.png`.
"""

from __future__ import annotations
import os
import sys
import tempfile
import time

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
    from scratch_optimization_engine.core.path_planner import (
        OccupancyGrid,
        AStarPlanner,
        TrajectorySmoother,
        CatmullRomSpline,
        resample_path,
    )
except (ModuleNotFoundError, ImportError):
    from core.path_planner import (
        OccupancyGrid,
        AStarPlanner,
        TrajectorySmoother,
        CatmullRomSpline,
        resample_path,
    )


def compute_path_length(path: np.ndarray) -> float:
    """Compute total cumulative Euclidean path length."""
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def compute_discrete_curvature(path: np.ndarray) -> np.ndarray:
    """
    Compute discrete curvature at interior vertices using Menger curvature formula:
    $$\\kappa = \\frac{2 \\sin(\\theta)}{\\|p_{i+1} - p_{i-1}\\|}$$
    """
    N = len(path)
    if N < 3:
        return np.zeros(N)

    curvatures = np.zeros(N)
    for i in range(1, N - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]

        v1 = a - b
        v2 = c - b
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        l3 = np.linalg.norm(c - a)

        if l1 < 1e-6 or l2 < 1e-6 or l3 < 1e-6:
            continue

        # Area of triangle formed by 3 points via cross product
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        # Menger curvature = 4 * Area / (l1 * l2 * l3)
        curv = (4.0 * area) / (l1 * l2 * l3)
        curvatures[i] = curv

    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    return curvatures


def run_path_planning_benchmark() -> None:
    """Execute complete discrete-continuous path planning and smoothing benchmark."""
    print("=" * 80)
    print("  SCRATCH OPTIMIZATION ENGINE: A* SEARCH & TRAJECTORY SMOOTHING BENCHMARK   ")
    print("=" * 80)

    # 1. Initialize 2D Occupancy Grid (100x100 resolution 1.0)
    width, height = 100, 100
    grid = OccupancyGrid(width=width, height=height, resolution=1.0, origin=(0.0, 0.0))

    # Add geometric obstacles creating corridors and detours
    # Wall 1: Vertical baffle with passage
    grid.add_box_obstacle(x_min=25.0, y_min=0.0, x_max=30.0, y_max=65.0)

    # Wall 2: Upper barrier
    grid.add_box_obstacle(x_min=50.0, y_min=35.0, x_max=55.0, y_max=100.0)

    # Circular obstacles
    grid.add_circular_obstacle(center_x=40.0, center_y=25.0, radius=7.0)
    grid.add_circular_obstacle(center_x=75.0, center_y=60.0, radius=10.0)
    grid.add_circular_obstacle(center_x=70.0, center_y=20.0, radius=8.0)

    # Wall 3: Lower right corridor guide
    grid.add_box_obstacle(x_min=75.0, y_min=32.0, x_max=95.0, y_max=36.0)

    start_pos = (8.0, 15.0)
    goal_pos = (92.0, 85.0)

    print(f"\nGrid Environment: {width}x{height} cells, Resolution: {grid.resolution}m")
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    print(f"Extracted Obstacle Boundary Points: {len(grid.obstacle_points)}")

    # 2. Discrete A* Planning
    print("\n[1/3] Executing Discrete A* Graph Search...")
    t_start_astar = time.perf_counter()
    planner = AStarPlanner(grid_map=grid, allow_diagonal=True)
    discrete_path = planner.plan(start_pos, goal_pos)
    t_elapsed_astar = (time.perf_counter() - t_start_astar) * 1000.0

    if discrete_path is None:
        raise RuntimeError("A* failed to find a valid path through the obstacle grid!")

    print(f"  A* Path Found: {len(discrete_path)} waypoints in {t_elapsed_astar:.2f} ms")

    # 3. Continuous Coordinate Gradient-Based Trajectory Smoothing
    print("\n[2/3] Executing Continuous Trajectory Smoothing (Adam Optimizer)...")
    num_control_pts = 65
    smoother = TrajectorySmoother(
        grid_map=grid,
        w_smooth=2.5,
        w_curv=0.8,
        w_obs=6.0,
        w_orig=0.15,
        d_safe=5.0,
        optimizer="adam",
        learning_rate=0.08,
        max_iter=350,
        tol=1e-5,
    )

    t_start_smooth = time.perf_counter()
    smoothed_waypoints = smoother.smooth(discrete_path, num_resample=num_control_pts)
    t_elapsed_smooth = (time.perf_counter() - t_start_smooth) * 1000.0
    print(f"  Smoothing Complete: {len(smoother.loss_history_)} iterations in {t_elapsed_smooth:.2f} ms")

    # 4. Continuous Catmull-Rom Spline Trajectory Generation
    print("\n[3/3] Evaluating C^1 Continuous Catmull-Rom Spline Trajectory...")
    spline = CatmullRomSpline(smoothed_waypoints)
    spline_pos, spline_vel, spline_acc, spline_curv = spline.evaluate(num_points_per_seg=20)
    print(f"  Spline Trajectory Evaluated: {len(spline_pos)} dense trajectory points")

    # 5. Quantitative Metric Calculations
    discrete_resampled = resample_path(discrete_path, len(spline_pos))
    discrete_dists, _ = grid.get_closest_obstacles(discrete_resampled)
    discrete_curv = compute_discrete_curvature(discrete_resampled)

    smoothed_dists, _ = grid.get_closest_obstacles(smoothed_waypoints)
    smoothed_curv = compute_discrete_curvature(smoothed_waypoints)

    spline_dists, _ = grid.get_closest_obstacles(spline_pos)

    len_discrete = compute_path_length(discrete_path)
    len_smoothed = compute_path_length(smoothed_waypoints)
    len_spline = compute_path_length(spline_pos)

    # Integral squared curvature: sum(kappa^2 * delta_s)
    ds_spline = np.sqrt(np.sum(np.diff(spline_pos, axis=0) ** 2, axis=1))
    int_curv_spline = float(np.sum((spline_curv[:-1] ** 2) * ds_spline))

    print("\n" + "=" * 76)
    print(f"{'Metric':<30} | {'Discrete A*':<14} | {'Smoothed':<12} | {'Spline Curve':<12}")
    print("-" * 76)
    print(f"{'Path Length (m)':<30} | {len_discrete:<14.2f} | {len_smoothed:<12.2f} | {len_spline:<12.2f}")
    print(f"{'Minimum Clearance (m)':<30} | {np.min(discrete_dists):<14.2f} | {np.min(smoothed_dists):<12.2f} | {np.min(spline_dists):<12.2f}")
    print(f"{'Mean Clearance (m)':<30} | {np.mean(discrete_dists):<14.2f} | {np.mean(smoothed_dists):<12.2f} | {np.mean(spline_dists):<12.2f}")
    print(f"{'Maximum Curvature (1/m)':<30} | {np.max(discrete_curv):<14.3f} | {np.max(smoothed_curv):<12.3f} | {np.max(spline_curv):<12.3f}")
    print(f"{'Integral Squared Curv':<30} | {'N/A':<14} | {'N/A':<12} | {int_curv_spline:<12.3f}")
    print("=" * 76)

    # 6. Publication-Quality Multi-Panel Diagnostic Plot
    print("\nGenerating Trajectory Optimization Diagnostic Visualization...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0])

    # Subplot 1: Main 2D Obstacle Grid and Trajectories
    ax1 = fig.add_subplot(gs[:, 0])

    # Compute Euclidean distance field heatmap on free space
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    grid_coords = np.column_stack([x_coords.ravel() + 0.5, y_coords.ravel() + 0.5])
    field_dists, _ = grid.get_closest_obstacles(grid_coords)
    dist_map = field_dists.reshape((height, width))
    dist_map[grid.grid] = 0.0

    # Display distance field background
    im = ax1.imshow(
        dist_map,
        origin="lower",
        cmap="Blues",
        extent=[0, width, 0, height],
        alpha=0.6,
        vmin=0.0,
        vmax=20.0,
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Obstacle Clearance Distance (m)", fontsize=11, fontweight="bold")

    # Render Obstacles in dark charcoal
    obs_overlay = np.ma.masked_where(~grid.grid, np.ones_like(grid.grid, dtype=float))
    ax1.imshow(
        obs_overlay,
        origin="lower",
        cmap="Greys",
        extent=[0, width, 0, height],
        alpha=0.9,
    )

    # Plot Paths
    ax1.plot(
        discrete_path[:, 0],
        discrete_path[:, 1],
        "r--s",
        markersize=3,
        linewidth=1.5,
        alpha=0.7,
        label=f"Discrete A* ({len_discrete:.1f} m)",
    )
    ax1.plot(
        smoothed_waypoints[:, 0],
        smoothed_waypoints[:, 1],
        "o-",
        color="#e67e22",
        markersize=4,
        linewidth=1.8,
        alpha=0.85,
        label=f"Smoothed Waypoints ({len_smoothed:.1f} m)",
    )
    ax1.plot(
        spline_pos[:, 0],
        spline_pos[:, 1],
        "-",
        color="#00ffff",
        linewidth=2.8,
        label=f"Catmull-Rom Spline ({len_spline:.1f} m)",
    )

    # Markers for Start and Goal
    ax1.scatter(
        start_pos[0],
        start_pos[1],
        color="#2ecc71",
        s=180,
        marker="o",
        edgecolors="black",
        linewidths=2,
        label=f"Start {start_pos}",
        zorder=10,
    )
    ax1.scatter(
        goal_pos[0],
        goal_pos[1],
        color="#f1c40f",
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label=f"Goal {goal_pos}",
        zorder=10,
    )

    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_xlabel("X World Coordinate (m)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Y World Coordinate (m)", fontsize=12, fontweight="bold")
    ax1.set_title("2D State-Space Obstacle Grid & Optimized Trajectory", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Subplot 2: Obstacle Clearance Profile
    ax2 = fig.add_subplot(gs[0, 1])
    s_spline = np.linspace(0.0, 1.0, len(spline_dists))
    s_discrete = np.linspace(0.0, 1.0, len(discrete_dists))

    ax2.plot(s_discrete, discrete_dists, "r--", linewidth=1.5, alpha=0.6, label="Discrete A*")
    ax2.plot(s_spline, spline_dists, color="#008080", linewidth=2.2, label="Spline Trajectory")
    ax2.axhline(smoother.d_safe, color="crimson", linestyle=":", linewidth=2.0, label=f"Safety Margin d_safe ({smoother.d_safe}m)")

    ax2.set_xlabel("Normalized Path Progress s in [0, 1]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Clearance Distance to Obstacle (m)", fontsize=11, fontweight="bold")
    ax2.set_title("Obstacle Clearance Profile", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    # Subplot 3: Curvature Profile kappa(s)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(s_discrete, discrete_curv, "r--", linewidth=1.2, alpha=0.6, label="Discrete Curvature")
    ax3.plot(s_spline, spline_curv, color="#8e44ad", linewidth=2.0, label="Spline Curvature kappa(s)")

    ax3.set_xlabel("Normalized Path Progress s in [0, 1]", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Curvature kappa (1/m)", fontsize=11, fontweight="bold")
    ax3.set_title("Trajectory Curvature Profile (Smoothing Effect)", fontsize=13, fontweight="bold")
    ax3.set_ylim(-0.02, min(0.6, max(np.max(spline_curv) * 1.5, 0.2)))
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "trajectory_visualization.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Trajectory visualization plot successfully saved to:\n  {output_path}\n")


if __name__ == "__main__":
    run_path_planning_benchmark()

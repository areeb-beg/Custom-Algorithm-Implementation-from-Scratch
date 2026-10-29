"""
Discrete A* State-Space Graph Search and Continuous Gradient-Based Trajectory Smoothing
======================================================================================

This module implements an end-to-end trajectory optimization pipeline:
1. `OccupancyGrid`: 2D binary spatial discretization with Euclidean distance transform.
2. `AStarPlanner`: Discrete 8-connected state-space shortest path planner with Octile heuristic.
3. `TrajectorySmoother`: Continuous coordinate gradient descent minimizing path curvature,
   elastic band tension, and obstacle potential clearance field using custom first-order optimizers.
4. `CatmullRomSpline`: $C^1$-continuous cubic spline interpolator with analytical velocity,
   acceleration, and curvature evaluations.

Mathematical Complexity:
------------------------
- A* Graph Search: Time $O(|V| \\log |V| + |E|)$, Space $O(|V|)$ where $|V| = H \\times W$.
- Trajectory Smoothing: Time $O(K \\cdot (N \\cdot M))$ where $K$ is optimizer iterations,
  $N$ is number of trajectory waypoints, and $M$ is number of obstacle boundary cells.
  Space $O(N + M)$.
- Spline Interpolation: Time $O(N_{\\text{dense}})$ evaluation time, Space $O(N_{\\text{dense}})$.
"""

from __future__ import annotations
import heapq
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scratch_optimization_engine.core.optimizers import Optimizer, Adam, SGD
except (ModuleNotFoundError, ImportError):
    from core.optimizers import Optimizer, Adam, SGD


class OccupancyGrid:
    """
    2D Binary Occupancy Grid and Euclidean Distance Field.

    Parameters
    ----------
    width : int
        Number of grid cells along the x-axis (columns).
    height : int
        Number of grid cells along the y-axis (rows).
    resolution : float
        Physical world metric size per grid cell (default: 1.0).
    origin : Tuple[float, float]
        World coordinates $(x_0, y_0)$ of grid cell (0, 0) (default: (0.0, 0.0)).

    Attributes
    ----------
    grid : np.ndarray
        2D boolean array of shape $(H, W)$, where True/1 indicates an obstacle.
    obstacle_points : np.ndarray
        Array of shape $(M, 2)$ containing world coordinates of obstacle boundary cells.
    """

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({width}, {height})")
        if resolution <= 0.0:
            raise ValueError(f"Resolution must be positive, got {resolution}")

        self.width: int = int(width)
        self.height: int = int(height)
        self.resolution: float = float(resolution)
        self.origin: tuple[float, float] = (float(origin[0]), float(origin[1]))

        self.grid: np.ndarray = np.zeros((self.height, self.width), dtype=bool)
        self.obstacle_points: np.ndarray = np.empty((0, 2), dtype=np.float64)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert continuous world coordinates $(x, y)$ to integer grid coordinates $(row, col)$."""
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor((y - self.origin[1]) / self.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert integer grid coordinates $(row, col)$ to continuous cell-center world coordinates $(x, y)$."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if grid cell $(row, col)$ is within map boundaries."""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int) -> bool:
        """Check if grid cell $(row, col)$ is an obstacle or out-of-bounds."""
        if not self.is_in_bounds(row, col):
            return True
        return bool(self.grid[row, col])

    def add_box_obstacle(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        """Add an axis-aligned rectangular obstacle in world coordinates."""
        r_min, c_min = self.world_to_grid(x_min, y_min)
        r_max, c_max = self.world_to_grid(x_max, y_max)

        r_start = max(0, min(r_min, r_max))
        r_end = min(self.height, max(r_min, r_max) + 1)
        c_start = max(0, min(c_min, c_max))
        c_end = min(self.width, max(c_min, c_max) + 1)

        self.grid[r_start:r_end, c_start:c_end] = True
        self._update_obstacle_points()

    def add_circular_obstacle(self, center_x: float, center_y: float, radius: float) -> None:
        """Add a circular obstacle in world coordinates."""
        r_center, c_center = self.world_to_grid(center_x, center_y)
        cell_radius = int(math.ceil(radius / self.resolution))

        r_start = max(0, r_center - cell_radius)
        r_end = min(self.height, r_center + cell_radius + 1)
        c_start = max(0, c_center - cell_radius)
        c_end = min(self.width, c_center + cell_radius + 1)

        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                wx, wy = self.grid_to_world(r, c)
                if (wx - center_x) ** 2 + (wy - center_y) ** 2 <= radius ** 2:
                    self.grid[r, c] = True

        self._update_obstacle_points()

    def _update_obstacle_points(self) -> None:
        """
        Extract obstacle boundary coordinates in world space.
        Boundary cells are obstacle cells adjacent to at least one free cell.
        """
        if not np.any(self.grid):
            self.obstacle_points = np.empty((0, 2), dtype=np.float64)
            return

        # Vectorized boundary extraction via morphological gradient
        padded = np.pad(self.grid, pad_width=1, mode="constant", constant_values=False)
        interior = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = self.grid & (~interior)

        rows, cols = np.nonzero(boundary)
        xs = self.origin[0] + (cols + 0.5) * self.resolution
        ys = self.origin[1] + (rows + 0.5) * self.resolution
        self.obstacle_points = np.column_stack((xs, ys)).astype(np.float64)

    def get_closest_obstacles(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Euclidean distances and nearest obstacle vectors for an array of points.

        Parameters
        ----------
        points : np.ndarray
            Array of coordinates of shape $(N, 2)$.

        Returns
        -------
        distances : np.ndarray
            Euclidean distance to closest obstacle point for each point, shape $(N,)$.
        closest_pts : np.ndarray
            Coordinates of closest obstacle point for each query point, shape $(N, 2)$.
        """
        points = np.asarray(points, dtype=np.float64)
        if len(self.obstacle_points) == 0:
            inf_dist = np.full(points.shape[0], 1e6, dtype=np.float64)
            return inf_dist, points.copy()

        # Vectorized pairwise Euclidean distance computation
        # points: (N, 2), obstacle_points: (M, 2)
        diff = points[:, np.newaxis, :] - self.obstacle_points[np.newaxis, :, :]  # (N, M, 2)
        dists_sq = np.sum(diff ** 2, axis=-1)  # (N, M)
        min_indices = np.argmin(dists_sq, axis=-1)  # (N,)

        closest_pts = self.obstacle_points[min_indices]
        distances = np.sqrt(np.take_along_axis(dists_sq, min_indices[:, np.newaxis], axis=1).squeeze(axis=1))

        return distances, closest_pts


class AStarPlanner:
    """
    Discrete 8-Connected A* Graph Search Algorithm.

    Guarantees admissible and consistent path search avoiding obstacle collision
    and diagonal corner-cutting.

    Mathematical Heuristic:
    -----------------------
    Octile Distance:
    $$h(n) = \\Delta x + \\Delta y + (\\sqrt{2} - 2) \\cdot \\min(\\Delta x, \\Delta y)$$
    where $\\Delta x = |x - x_g|, \\Delta y = |y - y_g|$.

    Parameters
    ----------
    grid_map : OccupancyGrid
        Underlying discrete occupancy grid.
    allow_diagonal : bool
        If True, permits 8-connected diagonal movements with cost $\\sqrt{2}$ (default: True).

    Complexity:
    -----------
    - Time: $O(|V| \\log |V| + |E|)$ where $|V| = H \\times W$ and $|E| \\le 8 |V|$.
    - Space: $O(|V|)$ memory for priority queue, distance map, and came-from parents.
    """

    def __init__(self, grid_map: OccupancyGrid, allow_diagonal: bool = True) -> None:
        self.grid_map: OccupancyGrid = grid_map
        self.allow_diagonal: bool = bool(allow_diagonal)

        # Movement offsets: (d_row, d_col, cost)
        self.motions: list[tuple[int, int, float]] = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
        ]
        if self.allow_diagonal:
            diag_cost = math.sqrt(2.0)
            self.motions.extend([
                (-1, -1, diag_cost),
                (-1, 1, diag_cost),
                (1, -1, diag_cost),
                (1, 1, diag_cost),
            ])

    def _heuristic(self, r: int, c: int, goal_r: int, goal_c: int) -> float:
        """Compute octile or Euclidean heuristic distance."""
        dr = abs(r - goal_r)
        dc = abs(c - goal_c)
        if self.allow_diagonal:
            # Octile distance
            return (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)
        else:
            # Manhattan distance
            return float(dr + dc)

    def plan(
        self, start_world: tuple[float, float], goal_world: tuple[float, float]
    ) -> Optional[np.ndarray]:
        """
        Execute A* path planning from start to goal in continuous world coordinates.

        Parameters
        ----------
        start_world : Tuple[float, float]
            Continuous start position $(x_s, y_s)$.
        goal_world : Tuple[float, float]
            Continuous goal position $(x_g, y_g)$.

        Returns
        -------
        Optional[np.ndarray]
            Array of waypoints $\\in \\mathbb{R}^{M \\times 2}$ from start to goal,
            or None if no collision-free path exists.
        """
        start_r, start_c = self.grid_map.world_to_grid(*start_world)
        goal_r, goal_c = self.grid_map.world_to_grid(*goal_world)

        if not self.grid_map.is_in_bounds(start_r, start_c) or self.grid_map.is_occupied(start_r, start_c):
            return None
        if not self.grid_map.is_in_bounds(goal_r, goal_c) or self.grid_map.is_occupied(goal_r, goal_c):
            return None

        # Priority queue stores tuples: (f_score, tie_breaker, r, c)
        tie_breaker = 0
        open_set: list[tuple[float, int, int, int]] = []
        heapq.heappush(open_set, (self._heuristic(start_r, start_c, goal_r, goal_c), tie_breaker, start_r, start_c))

        g_score: dict[tuple[int, int], float] = {(start_r, start_c): 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        closed_set: set[tuple[int, int]] = set()

        while open_set:
            f, _, curr_r, curr_c = heapq.heappop(open_set)
            curr = (curr_r, curr_c)

            if curr in closed_set:
                continue
            closed_set.add(curr)

            if curr_r == goal_r and curr_c == goal_c:
                # Reconstruct path
                path_grid = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path_grid.append(curr)
                path_grid.reverse()

                # Convert to world coordinates
                waypoints = [start_world]
                for r, c in path_grid[1:-1]:
                    waypoints.append(self.grid_map.grid_to_world(r, c))
                waypoints.append(goal_world)

                return np.array(waypoints, dtype=np.float64)

            curr_g = g_score[curr]

            for dr, dc, cost in self.motions:
                nr, nc = curr_r + dr, curr_c + dc
                neighbor = (nr, nc)

                if not self.grid_map.is_in_bounds(nr, nc) or self.grid_map.is_occupied(nr, nc):
                    continue

                # Prevent diagonal corner-cutting
                if dr != 0 and dc != 0:
                    if self.grid_map.is_occupied(curr_r, nc) or self.grid_map.is_occupied(nr, curr_c):
                        continue

                tentative_g = curr_g + cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g
                    h = self._heuristic(nr, nc, goal_r, goal_c)
                    tie_breaker += 1
                    # Small epsilon tie-breaking towards goal
                    f_score = tentative_g + h * 1.0001
                    heapq.heappush(open_set, (f_score, tie_breaker, nr, nc))

        return None


def resample_path(waypoints: np.ndarray, num_points: int) -> np.ndarray:
    """
    Uniformly resample trajectory waypoints along cumulative Euclidean arc-length.

    Parameters
    ----------
    waypoints : np.ndarray
        Array of path coordinates $(N, 2)$.
    num_points : int
        Desired number of equidistant waypoints.

    Returns
    -------
    np.ndarray
        Resampled waypoints $(num_points, 2)$.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) <= 2 or num_points <= len(waypoints):
        if len(waypoints) == num_points:
            return waypoints.copy()

    # Cumulative arc-length parameterization
    diffs = np.diff(waypoints, axis=0)
    seg_lens = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_dist = cum_dist[-1]

    if total_dist == 0.0:
        return np.repeat(waypoints[:1], num_points, axis=0)

    target_distances = np.linspace(0.0, total_dist, num_points)
    resampled = np.empty((num_points, 2), dtype=np.float64)

    # 1D linear interpolation across cumulative distances
    resampled[:, 0] = np.interp(target_distances, cum_dist, waypoints[:, 0])
    resampled[:, 1] = np.interp(target_distances, cum_dist, waypoints[:, 1])

    return resampled


class TrajectorySmoother:
    """
    Continuous Coordinate Gradient Descent Trajectory Optimizer.

    Optimizes interior waypoints $P = [p_1, \\dots, p_{N-1}]^T$ while clamping
    boundary endpoints $p_0$ and $p_N$ to start and goal positions.

    Composite Objective Formulation:
    --------------------------------
    $$J(P) = w_{\\text{smooth}} J_{\\text{smooth}} + w_{\\text{curv}} J_{\\text{curv}} + w_{\\text{obs}} J_{\\text{obs}} + w_{\\text{orig}} J_{\\text{orig}}$$

    1. Smoothness (Biharmonic / Minimum Acceleration):
       $$J_{\\text{smooth}} = \\frac{1}{2} \\sum_{i=1}^{N-1} \\|p_{i+1} - 2p_i + p_{i-1}\\|_2^2$$
    2. Curvature / Tension (Elastic Band):
       $$J_{\\text{curv}} = \\frac{1}{2} \\sum_{i=0}^{N-1} \\|p_{i+1} - p_i\\|_2^2$$
    3. Obstacle Repulsive Potential:
       $$U_{\\text{obs}}(p_i) = \\frac{1}{2} \\left( \\frac{1}{d(p_i)} - \\frac{1}{d_{\\text{safe}}} \\right)^2 \\quad \\forall d(p_i) < d_{\\text{safe}}$$
    4. Waypoint Anchor:
       $$J_{\\text{orig}} = \\frac{1}{2} \\sum_{i=1}^{N-1} \\|p_i - p_{i,\\text{orig}}\\|_2^2$$

    Parameters
    ----------
    grid_map : OccupancyGrid
        Occupancy grid containing obstacle boundaries.
    w_smooth : float
        Smoothness weight (acceleration minimization) (default: 1.5).
    w_curv : float
        Tension weight (arc-length / tension minimization) (default: 0.5).
    w_obs : float
        Obstacle repulsion weight (default: 3.0).
    w_orig : float
        Anchor weight to initial path (default: 0.2).
    d_safe : float
        Safety clearance threshold in world units (default: 2.0).
    optimizer : Union[Optimizer, str]
        First-order optimizer instance or string ('adam', 'momentum', 'sgd') (default: 'adam').
    learning_rate : float
        Gradient descent step size (default: 0.05).
    max_iter : int
        Maximum gradient optimization steps (default: 300).
    tol : float
        Convergence tolerance on coordinate updates (default: 1e-5).

    Complexity:
    -----------
    - Time: $O(K \\cdot N \\cdot M)$ where $K$ is iterations, $N$ is waypoints, and $M$ is obstacle count.
    - Space: $O(N \\cdot 2)$ for parameter buffers.
    """

    def __init__(
        self,
        grid_map: OccupancyGrid,
        w_smooth: float = 1.5,
        w_curv: float = 0.5,
        w_obs: float = 3.0,
        w_orig: float = 0.2,
        d_safe: float = 2.0,
        optimizer: Union[Optimizer, str] = "adam",
        learning_rate: float = 0.05,
        max_iter: int = 300,
        tol: float = 1e-5,
    ) -> None:
        self.grid_map: OccupancyGrid = grid_map
        self.w_smooth: float = float(w_smooth)
        self.w_curv: float = float(w_curv)
        self.w_obs: float = float(w_obs)
        self.w_orig: float = float(w_orig)
        self.d_safe: float = float(d_safe)
        self.learning_rate: float = float(learning_rate)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)

        if isinstance(optimizer, str):
            self.optimizer: Optimizer = Adam(learning_rate=self.learning_rate) if optimizer == "adam" else SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            self.optimizer = optimizer

        self.loss_history_: list[float] = []

    def compute_objective_and_gradient(
        self, P: np.ndarray, P_orig: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Evaluate composite loss $J(P)$ and analytical gradient $\\nabla_{P[1:N]} J(P)$.

        Parameters
        ----------
        P : np.ndarray
            Current complete path array $(N+1, 2)$.
        P_orig : np.ndarray
            Initial reference path array $(N+1, 2)$.

        Returns
        -------
        loss : float
            Scalar objective loss value.
        grad_interior : np.ndarray
            Gradient w.r.t interior waypoints $P[1:N]$, shape $(N-1, 2)$.
        """
        N_pts = len(P)
        grad_interior = np.zeros((N_pts - 2, 2), dtype=np.float64)
        total_loss = 0.0

        # 1. Smoothness term (minimum acceleration): J_smooth = 0.5 * sum ||p_{i+1} - 2p_i + p_{i-1}||^2
        if self.w_smooth > 0.0:
            # Second differences: a_k = p_{k+2} - 2p_{k+1} + p_k for k=0...N_pts-3
            # corresponding to indices i=1...N_pts-2
            A = P[2:] - 2.0 * P[1:-1] + P[:-2]  # shape (N_pts - 2, 2)
            j_smooth = 0.5 * np.sum(A ** 2)
            total_loss += self.w_smooth * j_smooth

            # Exact analytical gradient w.r.t interior points P[1:-1]:
            grad_smooth = -2.0 * A
            if len(A) > 1:
                grad_smooth[:-1] += A[1:]
                grad_smooth[1:] += A[:-1]
            grad_interior += self.w_smooth * grad_smooth

        # 2. Curvature / Tension term: J_curv = 0.5 * sum ||p_{i+1} - p_i||^2
        if self.w_curv > 0.0:
            diffs = P[1:] - P[:-1]  # shape (N_pts - 1, 2)
            j_curv = 0.5 * np.sum(diffs ** 2)
            total_loss += self.w_curv * j_curv

            # grad w.r.t p_i: 2p_i - p_{i-1} - p_{i+1}
            grad_curv = 2.0 * P[1:-1] - P[:-2] - P[2:]
            grad_interior += self.w_curv * grad_curv

        # 3. Obstacle Repulsion term
        if self.w_obs > 0.0 and len(self.grid_map.obstacle_points) > 0:
            interior_pts = P[1:-1]
            dists, closest_pts = self.grid_map.get_closest_obstacles(interior_pts)

            # Mask points within safety clearance
            active_mask = dists < self.d_safe
            if np.any(active_mask):
                active_d = np.maximum(dists[active_mask], 1e-4)
                repulsion_mag = (1.0 / active_d) - (1.0 / self.d_safe)
                j_obs = 0.5 * np.sum(repulsion_mag ** 2)
                total_loss += self.w_obs * j_obs

                # Analytical gradient:
                # nabla U(p) = - (1/d - 1/d_safe) * (1/d^2) * (p - o*) / d
                direction = interior_pts[active_mask] - closest_pts[active_mask]
                grad_obs_active = -(repulsion_mag / (active_d ** 3))[:, np.newaxis] * direction
                grad_interior[active_mask] += self.w_obs * grad_obs_active

        # 4. Waypoint Anchor term: J_orig = 0.5 * sum ||p_i - p_{i,orig}||^2
        if self.w_orig > 0.0:
            anchor_diff = P[1:-1] - P_orig[1:-1]
            j_orig = 0.5 * np.sum(anchor_diff ** 2)
            total_loss += self.w_orig * j_orig
            grad_interior += self.w_orig * anchor_diff

        return total_loss, grad_interior

    def smooth(self, raw_waypoints: np.ndarray, num_resample: int = 50) -> np.ndarray:
        """
        Execute continuous coordinate gradient descent trajectory smoothing.

        Parameters
        ----------
        raw_waypoints : np.ndarray
            Discrete input waypoints $(N, 2)$ from path planner.
        num_resample : int
            Number of points for equidistant initial trajectory resampling (default: 50).

        Returns
        -------
        np.ndarray
            Optimized, smoothed path array $(num_resample, 2)$.
        """
        P = resample_path(raw_waypoints, num_resample)
        P_orig = P.copy()
        self.optimizer.reset()
        self.loss_history_ = []

        for step in range(self.max_iter):
            loss, grad = self.compute_objective_and_gradient(P, P_orig)
            self.loss_history_.append(loss)

            # Gradient update on free interior waypoints
            updated_interior = self.optimizer.step(P[1:-1], grad)

            # Clamp coordinates to grid boundaries
            max_x = self.grid_map.origin[0] + self.grid_map.width * self.grid_map.resolution
            max_y = self.grid_map.origin[1] + self.grid_map.height * self.grid_map.resolution
            updated_interior[:, 0] = np.clip(updated_interior[:, 0], self.grid_map.origin[0], max_x)
            updated_interior[:, 1] = np.clip(updated_interior[:, 1], self.grid_map.origin[1], max_y)

            delta = np.max(np.abs(updated_interior - P[1:-1]))
            P[1:-1] = updated_interior

            if delta < self.tol and step > 15:
                break

        return P


class CatmullRomSpline:
    """
    $C^1$-Continuous Catmull-Rom Cubic Spline Trajectory Generator.

    Generates smooth, continuously differentiable trajectories passing directly
    through control points with analytical velocity, acceleration, and curvature.

    Mathematical Formulation:
    -------------------------
    For segment $i$ between $P_i$ and $P_{i+1}$ with parameter $u \\in [0, 1]$:
    $$C(u) = (2u^3 - 3u^2 + 1) P_i + (u^3 - 2u^2 + u) T_i + (-2u^3 + 3u^2) P_{i+1} + (u^3 - u^2) T_{i+1}$$
    where tangents $T_i = \\frac{1}{2} (P_{i+1} - P_{i-1})$.

    Curvature Formula:
    ------------------
    $$\\kappa(u) = \\frac{|x'(u) y''(u) - y'(u) x''(u)|}{\\left(x'(u)^2 + y'(u)^2\\right)^{3/2}}$$

    Complexity:
    -----------
    - Time: $O(M)$ evaluation operations for $M$ interpolated points.
    - Space: $O(M)$ output arrays.
    """

    def __init__(self, waypoints: np.ndarray) -> None:
        self.waypoints: np.ndarray = np.asarray(waypoints, dtype=np.float64)
        if len(self.waypoints) < 3:
            raise ValueError("CatmullRomSpline requires at least 3 waypoints")

        # Pad endpoints to compute boundary tangents: P_{-1} = 2 P_0 - P_1, P_{N+1} = 2 P_N - P_{N-1}
        p0_ext = 2.0 * self.waypoints[0] - self.waypoints[1]
        pn_ext = 2.0 * self.waypoints[-1] - self.waypoints[-2]
        self.pts: np.ndarray = np.vstack([p0_ext, self.waypoints, pn_ext])

    def evaluate(self, num_points_per_seg: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate spline curve position, velocity, acceleration, and curvature.

        Parameters
        ----------
        num_points_per_seg : int
            Number of sample points evaluated per spline segment.

        Returns
        -------
        positions : np.ndarray
            Trajectory positions $(K, 2)$.
        velocities : np.ndarray
            First derivative tangent vectors $(K, 2)$.
        accelerations : np.ndarray
            Second derivative acceleration vectors $(K, 2)$.
        curvatures : np.ndarray
            Analytical scalar curvature profile $(K,)$.
        """
        n_segs = len(self.waypoints) - 1
        u = np.linspace(0.0, 1.0, num_points_per_seg, endpoint=False)

        # Hermite basis functions
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2

        # First derivatives
        dh00 = 6.0 * u**2 - 6.0 * u
        dh10 = 3.0 * u**2 - 4.0 * u + 1.0
        dh01 = -6.0 * u**2 + 6.0 * u
        dh11 = 3.0 * u**2 - 2.0 * u

        # Second derivatives
        ddh00 = 12.0 * u - 6.0
        ddh10 = 6.0 * u - 4.0
        ddh01 = -12.0 * u + 6.0
        ddh11 = 6.0 * u - 2.0

        all_pos = []
        all_vel = []
        all_acc = []

        for i in range(1, n_segs + 1):
            p_prev = self.pts[i - 1]
            p_curr = self.pts[i]
            p_next = self.pts[i + 1]
            p_next2 = self.pts[i + 2]

            t_curr = 0.5 * (p_next - p_prev)
            t_next = 0.5 * (p_next2 - p_curr)

            # Vectorized evaluation for current segment
            pos_seg = (
                h00[:, np.newaxis] * p_curr
                + h10[:, np.newaxis] * t_curr
                + h01[:, np.newaxis] * p_next
                + h11[:, np.newaxis] * t_next
            )
            vel_seg = (
                dh00[:, np.newaxis] * p_curr
                + dh10[:, np.newaxis] * t_curr
                + dh01[:, np.newaxis] * p_next
                + dh11[:, np.newaxis] * t_next
            )
            acc_seg = (
                ddh00[:, np.newaxis] * p_curr
                + ddh10[:, np.newaxis] * t_curr
                + ddh01[:, np.newaxis] * p_next
                + ddh11[:, np.newaxis] * t_next
            )

            all_pos.append(pos_seg)
            all_vel.append(vel_seg)
            all_acc.append(acc_seg)

        # Include exact final waypoint
        all_pos.append(self.waypoints[-1:])
        all_vel.append(all_vel[-1][-1:])
        all_acc.append(all_acc[-1][-1:])

        positions = np.vstack(all_pos)
        velocities = np.vstack(all_vel)
        accelerations = np.vstack(all_acc)

        # Analytical Curvature: kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
        xp = velocities[:, 0]
        yp = velocities[:, 1]
        xpp = accelerations[:, 0]
        ypp = accelerations[:, 1]

        numerator = np.abs(xp * ypp - yp * xpp)
        denominator = np.maximum((xp**2 + yp**2) ** 1.5, 1e-9)
        curvatures = numerator / denominator

        return positions, velocities, accelerations, curvatures

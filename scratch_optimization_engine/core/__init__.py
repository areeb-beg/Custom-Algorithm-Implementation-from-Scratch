"""
Scratch Optimization Engine - Core Library Modules
==================================================

Pure dependency-free numerical optimization and machine learning library
implemented from mathematical first principles using only NumPy and Matplotlib.
"""

from scratch_optimization_engine.core.optimizers import (
    Optimizer,
    SGD,
    RMSProp,
    Adam,
    get_optimizer,
)

from scratch_optimization_engine.core.regression import (
    PolynomialFeatures,
    MultivariatePolynomialRegression,
    soft_thresholding,
    cholesky_solve,
)

from scratch_optimization_engine.core.path_planner import (
    OccupancyGrid,
    AStarPlanner,
    TrajectorySmoother,
    CatmullRomSpline,
    resample_path,
)

__all__ = [
    "Optimizer",
    "SGD",
    "RMSProp",
    "Adam",
    "get_optimizer",
    "PolynomialFeatures",
    "MultivariatePolynomialRegression",
    "soft_thresholding",
    "cholesky_solve",
    "OccupancyGrid",
    "AStarPlanner",
    "TrajectorySmoother",
    "CatmullRomSpline",
    "resample_path",
]

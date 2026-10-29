"""
Scratch Optimization Engine
===========================
Dependency-free numerical optimization and machine learning library in Python
using only NumPy and Matplotlib.
"""

from scratch_optimization_engine.core import (
    Optimizer,
    SGD,
    RMSProp,
    Adam,
    get_optimizer,
    PolynomialFeatures,
    MultivariatePolynomialRegression,
    soft_thresholding,
    cholesky_solve,
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

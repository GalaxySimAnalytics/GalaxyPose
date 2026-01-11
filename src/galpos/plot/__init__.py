"""Plotting submodule (optional dependency).

This submodule requires `matplotlib`. The core `galpos` package does *not* depend on
matplotlib, so importing `galpos` should work without it.

Install plotting support with:

- `pip install matplotlib`
"""

from __future__ import annotations

from typing import Any


_MATPLOTLIB_INSTALL_HINT = "Plotting requires 'matplotlib'. Install with: pip install matplotlib"


def _import_pose_trajectory() -> tuple[Any, Any]:
    try:
        from .pose_trajectory import plot_galaxy_pose_trajectory, plot_galaxy_pose_trajectory_3d
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or ""
        msg = str(e)
        if missing.startswith("matplotlib") or ("matplotlib" in msg):
            raise ImportError(_MATPLOTLIB_INSTALL_HINT) from e
        raise
    return plot_galaxy_pose_trajectory, plot_galaxy_pose_trajectory_3d


def _import_sfh() -> tuple[Any, Any]:
    try:
        from .sfh import plot_sfr_evolution, SFHPlotConfig
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or ""
        msg = str(e)
        if missing.startswith("matplotlib") or ("matplotlib" in msg):
            raise ImportError(_MATPLOTLIB_INSTALL_HINT) from e
        raise
    return plot_sfr_evolution, SFHPlotConfig


def __getattr__(name: str) -> object:
    if name in {"plot_galaxy_pose_trajectory", "plot_galaxy_pose_trajectory_3d"}:
        fn2d, fn3d = _import_pose_trajectory()
        return fn2d if name == "plot_galaxy_pose_trajectory" else fn3d
    if name in {"plot_sfr_evolution", "SFHPlotConfig"}:
        fn, cfg = _import_sfh()
        return fn if name == "plot_sfr_evolution" else cfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + [
        "plot_sfr_evolution",
        "SFHPlotConfig",
        "plot_galaxy_pose_trajectory",
        "plot_galaxy_pose_trajectory_3d",
    ])


__all__ = [
    "plot_sfr_evolution",
    "SFHPlotConfig",
    "plot_galaxy_pose_trajectory",
    "plot_galaxy_pose_trajectory_3d",
]
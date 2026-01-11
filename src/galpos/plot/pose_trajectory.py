"""
Visualization utilities for GalaxyPose trajectories.

This module provides convenience plotting helpers for inspecting a
:class:`~galpos.GalaxyPoseTrajectory`-like object.

Functions
---------
- ``plot_galaxy_pose_trajectory``: 2D diagnostic panels (1 row x 3 columns):
  (pos components, vel components, optional orientation axis components).
- ``plot_galaxy_pose_trajectory_3d``: 3D orbit plot with optional pose frames.

Time/orientation conventions
----------------------------
Rotation matrix conventions vary across projects. Here we use a practical,
plot-oriented convention:

- For a rotation matrix ``R`` evaluated from ``GalaxyPoseTrajectory.__call__``,
  we define the "disk axis" as the third column ``R[:, 2]`` (for each time),
  i.e. the image of the +z basis vector under ``R``.

If your internal convention differs (e.g., transpose or inverse), adapt
``axis_from_rotation`` and/or ``_axes_from_rotation`` accordingly.

Sampling visualization
----------------------
When ``show_sampling=True``, the original sampled points stored on the trajectory
(and orientation, if available) are over-plotted as markers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from galpos import GalaxyPoseTrajectory
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

def _as_1d_times(t: Optional[ArrayLike], t0: float, t1: float, n_samples: int) -> NDArray[np.floating]:
    """Normalize time input to a 1D float array."""
    if t is None:
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 when t is None.")
        return np.linspace(float(t0), float(t1), int(n_samples), dtype=float)

    tt = np.asarray(t, dtype=float)
    if tt.ndim == 0:
        tt = tt.reshape(1)
    if tt.ndim != 1:
        raise ValueError("t must be 1D array-like or scalar.")
    if tt.size == 0:
        raise ValueError("t must be non-empty.")
    return tt


def _component_labels(ndim: int) -> list[str]:
    """Human-friendly labels for vector components."""
    if ndim == 3:
        return ["x", "y", "z"]
    return [f"c{i}" for i in range(ndim)]


def axis_from_rotation(rot: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return axis vector(s) from rotation matrix array (third column).

    Parameters
    ----------
    rot
        Rotation matrices with shape (3, 3) or (M, 3, 3).

    Returns
    -------
    axis
        Axis vectors with shape (3,) or (M, 3).

    Raises
    ------
    ValueError
        If `rot` does not have shape (3,3) or (M,3,3).
    """
    if rot.ndim == 2:
        if rot.shape != (3, 3):
            raise ValueError(f"Expected rot shape (3,3), got {rot.shape}.")
        return rot[:, 2]
    if rot.ndim == 3:
        if rot.shape[1:] != (3, 3):
            raise ValueError(f"Expected rot shape (M,3,3), got {rot.shape}.")
        return rot[:, :, 2]
    raise ValueError(f"Expected rot with ndim 2 or 3, got ndim={rot.ndim}.")

def _as_2d(a: NDArray[np.floating]) -> NDArray[np.floating]:
    """Ensure (N, ndim) layout for possibly-1D arrays."""
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a

def _apply_clean_style(ax: Axes) -> None:
    """Apply a compact, publication-like style to an Axes (no global rcParams)."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out", length=3, width=0.8)


def _set_component_legend(ax: Axes, *, ncol: int = 3) -> None:
    """Small legend with minimal visual weight."""
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=int(ncol),
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.2,
        fontsize="small",
    )

def plot_galaxy_pose_trajectory(
    gpt: "GalaxyPoseTrajectory",
    t: Optional[ArrayLike] = None,
    *,
    n_samples: int = 256,
    wrap: bool = False,
    extrapolate: bool = False,
    show_orientation: bool = True,
    show_sampling: bool = False,
    figsize: Tuple[float, float] = (12.0, 3.6),
    dpi: int = 150,
) -> Tuple["Figure", Dict[str, "Axes"]]:
    """Visualize orbit components and (optionally) orientation evolution (panels).

    Parameters
    ----------
    gpt
        A :class:`~galpos.GalaxyPoseTrajectory`-like object.
    t
        Times to evaluate. If None, uses `n_samples` uniformly spaced points over
        the stored trajectory time span.
    n_samples
        Number of samples when `t is None`.
    wrap
        Passed to `gpt(t, wrap=..., ...)`.
    extrapolate
        Passed to `gpt(t, extrapolate=..., ...)`.
    show_orientation
        If True and orientation is available, plot the inferred axis evolution.
    show_sampling
        If True, over-plot the original sampled data points as markers using
        `gpt.trajectory.times/positions/velocities` and `gpt.orientation.times/rotations`.
    figsize
        Figure size in inches.
    dpi
        Figure DPI.

    Returns
    -------
    fig
        Matplotlib figure.
    axes
        Dict of axes. Always contains ``'pos'`` and ``'vel'``; contains ``'ori'``
        only when ``show_orientation=True``.

    Raises
    ------
    ValueError
        If evaluated `pos` or `vel` do not have shape (M, 3) or (M, ndim).
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or ""
        msg = str(e)
        if missing.startswith("matplotlib") or ("matplotlib" in msg):
            raise ImportError("Plotting requires 'matplotlib'. Install with: pip install matplotlib") from e
        raise

    if dpi <= 0:
        raise ValueError("dpi must be > 0.")

    traj_t = np.asarray(gpt.trajectory.times, dtype=float)
    if traj_t.ndim != 1 or traj_t.size < 2:
        raise ValueError("gpt.trajectory.times must be a 1D array with length >= 2.")
    tt = _as_1d_times(t, float(traj_t[0]), float(traj_t[-1]), n_samples)

    pos, vel, rot = gpt(tt, wrap=wrap, extrapolate=extrapolate)
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)

    if pos.ndim != 2:
        raise ValueError(f"Expected pos to be 2D (M,ndim), got {pos.shape}.")
    if vel.ndim != 2:
        raise ValueError(f"Expected vel to be 2D (M,ndim), got {vel.shape}.")
    if pos.shape[0] != tt.size or vel.shape[0] != tt.size:
        raise ValueError("pos/vel first dimension must match number of queried times.")
    if pos.shape[1] != vel.shape[1]:
        raise ValueError("pos and vel must have the same ndim.")
    ndim = int(pos.shape[1])
    labels = _component_labels(ndim)

    # consistent, simple colors for components
    comp_colors = ["C0", "C1", "C2", "C3", "C4"]

    ncols = 3 if show_orientation else 2
    fig, axs = plt.subplots(1, ncols, figsize=figsize, dpi=int(dpi), constrained_layout=True)

    ax_pos = axs[0]
    ax_vel = axs[1]
    ax_ori = axs[2] if show_orientation else None

    for ax in (ax_pos, ax_vel):
        _apply_clean_style(ax)
    if ax_ori is not None:
        _apply_clean_style(ax_ori)

    # --- pos components ---
    for j in range(ndim):
        ax_pos.plot(
            tt, pos[:, j],
            color=comp_colors[j % len(comp_colors)],
            lw=1.6,
            label=labels[j],
        )
    ax_pos.set_title("Position", pad=6)
    ax_pos.set_xlabel("t")
    ax_pos.set_ylabel("pos")
    _set_component_legend(ax_pos, ncol=min(3, ndim))

    # --- vel components ---
    for j in range(ndim):
        ax_vel.plot(
            tt, vel[:, j],
            color=comp_colors[j % len(comp_colors)],
            lw=1.6,
            label=f"v{labels[j]}",
        )
    ax_vel.set_title("Velocity", pad=6)
    ax_vel.set_xlabel("t")
    ax_vel.set_ylabel("vel")
    _set_component_legend(ax_vel, ncol=min(3, ndim))

    # --- optional sampling points (INIT data used to construct the object) ---
    if show_sampling:
        ts = np.asarray(gpt.trajectory.times, dtype=float)

        ps = _as_2d(np.asarray(gpt.trajectory.positions, dtype=float))
        if ts.ndim == 1 and ps.ndim == 2 and ps.shape[0] == ts.size:
            m = min(int(ps.shape[1]), int(ndim))
            for j in range(m):
                ax_pos.plot(
                    ts, ps[:, j],
                    linestyle="none",
                    marker="o",
                    ms=3.0,
                    alpha=0.55,
                    color=comp_colors[j % len(comp_colors)],
                    label="_nolegend_",
                )

        vs = gpt.trajectory.velocities
        if vs is not None:
            vs_arr = _as_2d(np.asarray(vs, dtype=float))
            if vs_arr.ndim == 2 and vs_arr.shape[0] == ts.size:
                m = min(int(vs_arr.shape[1]), int(ndim))
                for j in range(m):
                    ax_vel.plot(
                        ts, vs_arr[:, j],
                        linestyle="none",
                        marker="o",
                        ms=3.0,
                        alpha=0.55,
                        color=comp_colors[j % len(comp_colors)],
                        label="_nolegend_",
                    )

    axes: Dict[str, Axes] = {"pos": ax_pos, "vel": ax_vel}

    # --- orientation axis (third column) ---
    if show_orientation and ax_ori is not None:
        axes["ori"] = ax_ori

        if rot is not None:
            rot_arr = np.asarray(rot, dtype=float)
            axis = axis_from_rotation(rot_arr)
            if axis.ndim == 1:
                axis = axis.reshape(1, 3)
            if axis.shape[0] != tt.size or axis.shape[1] != 3:
                raise ValueError(f"Expected orientation axis shape (M,3), got {axis.shape}.")

            nrm = np.linalg.norm(axis, axis=1)
            good = nrm > 0
            axis_n = np.zeros_like(axis)
            axis_n[good] = axis[good] / nrm[good, None]

            ax_ori.plot(tt, axis_n[:, 0], color="C0", lw=1.6, label="nx")
            ax_ori.plot(tt, axis_n[:, 1], color="C1", lw=1.6, label="ny")
            ax_ori.plot(tt, axis_n[:, 2], color="C2", lw=1.6, label="nz")

            if show_sampling and gpt.orientation is not None:
                ot = np.asarray(gpt.orientation.times, dtype=float)
                orot = np.asarray(gpt.orientation.rotations, dtype=float)
                oaxis = axis_from_rotation(orot)
                if oaxis.ndim == 1:
                    oaxis = oaxis.reshape(1, 3)

                if ot.ndim == 1 and oaxis.ndim == 2 and oaxis.shape[0] == ot.size and oaxis.shape[1] == 3:
                    onrm = np.linalg.norm(oaxis, axis=1)
                    ogood = onrm > 0
                    oaxis_n = np.zeros_like(oaxis)
                    oaxis_n[ogood] = oaxis[ogood] / onrm[ogood, None]
                    ax_ori.plot(ot, oaxis_n[:, 0], linestyle="none", marker="o", 
                                ms=3.0, alpha=0.55, color="C0", label="_nolegend_")
                    ax_ori.plot(ot, oaxis_n[:, 1], linestyle="none", marker="o", 
                                ms=3.0, alpha=0.55, color="C1", label="_nolegend_")
                    ax_ori.plot(ot, oaxis_n[:, 2], linestyle="none", marker="o", 
                                ms=3.0, alpha=0.55, color="C2", label="_nolegend_")

            ax_ori.set_title("Orientation axis", pad=6)
            ax_ori.set_xlabel("t")
            ax_ori.set_ylabel("n (unit)")
            ax_ori.set_ylim(-1.05, 1.05)
            _set_component_legend(ax_ori, ncol=3)
        else:
            ax_ori.set_title("Orientation axis", pad=6)
            ax_ori.text(0.5, 0.5, "No orientation", ha="center", va="center", transform=ax_ori.transAxes)
            ax_ori.grid(False)
            ax_ori.spines["left"].set_visible(False)
            ax_ori.spines["bottom"].set_visible(False)
            ax_ori.set_xticks([])
            ax_ori.set_yticks([])

    return fig, axes


# -------------------- 3D orbit plotting --------------------

def _axes_from_rotation(
    rot: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return per-sample basis vectors (ex, ey, ez) from rotation matrices (columns)."""
    if rot.ndim == 2:
        if rot.shape != (3, 3):
            raise ValueError(f"Expected rot shape (3,3), got {rot.shape}.")
        ex = rot[:, 0].reshape(1, 3)
        ey = rot[:, 1].reshape(1, 3)
        ez = rot[:, 2].reshape(1, 3)
        return ex, ey, ez
    if rot.ndim == 3:
        if rot.shape[1:] != (3, 3):
            raise ValueError(f"Expected rot shape (M,3,3), got {rot.shape}.")
        return rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
    raise ValueError(f"Expected rot with ndim 2 or 3, got ndim={rot.ndim}.")


def plot_galaxy_pose_trajectory_3d(
    gpt: "GalaxyPoseTrajectory",
    t: Optional[ArrayLike] = None,
    *,
    n_samples: int = 256,
    wrap: bool = False,
    extrapolate: bool = False,
    show_poses: bool = True,
    show_sampling: bool = False,
    pose_stride: int = 16,
    pose_scale: Optional[float] = None,
    orbit_kwargs: Optional[Dict[str, object]] = None,
    figsize: Tuple[float, float] = (8.0, 7.0),
    dpi: int = 150,
    elev: float = 20.0,
    azim: float = -60.0,
) -> Tuple["Figure", "Axes"]:
    """Plot a 3D orbit and (optionally) pose frames along the orbit.

    Parameters
    ----------
    gpt
        A :class:`~galpos.GalaxyPoseTrajectory`-like object.
    t
        Times to evaluate. If None, sample uniformly in time with `n_samples`.
    n_samples
        Number of samples when `t is None`.
    wrap, extrapolate
        Passed to `gpt(...)`.
    show_poses
        If True and orientation is available, draw pose frames along the orbit.
    show_sampling
        If True, scatter original sampled orbit points from `gpt.trajectory`.
    pose_stride
        Draw one pose frame every `pose_stride` samples.
    pose_scale
        Length scale of pose axes in data units. If None, choose ~5% of orbit span.
    orbit_kwargs
        Keyword args for the orbit line (e.g., {'color': 'k', 'lw': 1.5}).
    figsize, dpi
        Figure size (inches) and DPI.
    elev, azim
        3D view angles.

    Returns
    -------
    fig
        Matplotlib Figure.
    ax
        A 3D Matplotlib Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or ""
        msg = str(e)
        if missing.startswith("matplotlib") or ("matplotlib" in msg):
            raise ImportError("Plotting requires 'matplotlib'. Install with: pip install matplotlib") from e
        raise

    if dpi <= 0:
        raise ValueError("dpi must be > 0.")
    if pose_stride <= 0:
        raise ValueError("pose_stride must be a positive integer.")

    traj_t = np.asarray(gpt.trajectory.times, dtype=float)
    if traj_t.ndim != 1 or traj_t.size < 2:
        raise ValueError("gpt.trajectory.times must be a 1D array with length >= 2.")
    tt = _as_1d_times(t, float(traj_t[0]), float(traj_t[-1]), n_samples)

    pos, _vel, rot = gpt(tt, wrap=wrap, extrapolate=extrapolate)
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"3D plot requires pos shape (M,3), got {pos.shape}.")

    if orbit_kwargs is None:
        orbit_kwargs = {"color": "0.2", "lw": 1.6}

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[arg-type]

    # orbit line
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], **orbit_kwargs)

    # start/end markers (clean, informative)
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], s=28, color="C0", depthshade=False, label="start")
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], s=28, color="C3", depthshade=False, label="end")

    # sampled orbit points (INIT data)
    if show_sampling:
        ps = np.asarray(gpt.trajectory.positions, dtype=float)
        if ps.ndim == 2 and ps.shape[1] == 3:
            ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=10, alpha=0.35, color="0.3", depthshade=False)

    # pose axis scale
    if pose_scale is None:
        span = np.nanmax(pos, axis=0) - np.nanmin(pos, axis=0)
        span_norm = float(np.linalg.norm(span))
        pose_scale = 0.05 * span_norm if np.isfinite(span_norm) and span_norm > 0 else 1.0

    # pose frames
    if show_poses and rot is not None:
        rot_arr = np.asarray(rot, dtype=float)
        ex, ey, ez = _axes_from_rotation(rot_arr)

        idx = np.arange(0, pos.shape[0], pose_stride, dtype=int)
        p = pos[idx]

        ax.quiver(p[:, 0], p[:, 1], p[:, 2], ex[idx, 0], ex[idx, 1], ex[idx, 2],
                  length=float(pose_scale), normalize=True, color="C0", linewidth=0.9, alpha=0.85)
        ax.quiver(p[:, 0], p[:, 1], p[:, 2], ey[idx, 0], ey[idx, 1], ey[idx, 2],
                  length=float(pose_scale), normalize=True, color="C1", linewidth=0.9, alpha=0.85)
        ax.quiver(p[:, 0], p[:, 1], p[:, 2], ez[idx, 0], ez[idx, 1], ez[idx, 2],
                  length=float(pose_scale), normalize=True, color="C2", linewidth=0.9, alpha=0.85)

    # labels + view
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_title("Orbit (3D)", pad=10)

    # cleaner 3D panes/grid
    try:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    except Exception:
        pass
    ax.grid(False)

    # equal-ish aspect (best effort across matplotlib versions)
    try:
        ax.set_box_aspect((1, 1, 1))  # mpl>=3.3
    except Exception:
        pass

    ax.legend(frameon=False, loc="upper left")

    return fig, ax
"""
Plotting tools for GalaxyPose.

This module currently provides `plot_sfr_evolution`, which produces a composite figure:
1) A star-formation history (SFH) "radial evolution" map, shown as a function of **lookback time**
   and cylindrical radius.
2) A grid of stellar-mass projection panels ("mass panels") binned in the same lookback-time
   convention and shown for three snapshots: current, birth-aligned, and birth-centered.

Time conventions
----------------
- `tform` is **cosmic time** of formation (Gyr), stored on birth snapshots (`birth_*`).
- The SFH axis is **lookback time** (Gyr), defined as::

    lookback = age_max - tform

  where `age_max` is the cosmic time at z=0 (or another reference "present-day" cosmic time).
- `SFHPlotConfig.t_range` is a lookback-time range `(t0, t1)` used consistently for:
  - SFH display limits (x-axis)
  - SFH binning range (passed through to the SFH builder)
  - mass-panel time binning (uniform bins across `t_range` unless explicit edges are provided)

Notes
-----
- This code assumes `current.s['age']` is a **lookback time** quantity in Gyr.
- `hist_2d` and `sfr_virus_radial_evolution` are imported from `.sfh_util`.
"""
from logging import warning
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, Union, TypedDict, List
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar

from .sfh_util import sfr_virus_radial_evolution, hist_2d, SCIENCEPLOT


if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap


@dataclass
class SFHPlotConfig:
    """Configuration for SFR evolution plot.

    Parameters
    ----------
    style
        Matplotlib rcParams to apply before drawing.
    sfr_color, mass_color
        Colormaps for the SFH map and mass panels, respectively.
    r_max
        Maximum radius used to derive the plotting region (kpc). If None, uses the 95th percentile
        of `current.s['r']`. The final radius is rounded up to a multiple of `r_round_to`.
    t_range
        Lookback-time range (Gyr) used for SFH display and time-binning.
    age_max
        Cosmic time (Gyr) for converting `tform` to lookback time: `lookback = age_max - tform`.
    n_time_panels
        Number of time-binned mass panels (excluding the first "Current" reference column).
    r_nbins, t_nbins
        SFH image resolution in R and t, respectively.
    nbins_xy
        2D histogram resolution for mass projections.

    Figure sizing (stable with changing n_time_panels)
    --------------------------------------------------
    fixed_figsize
        If set, uses a fixed `(width, height)` in inches for the whole figure.
    base_n_col_for_size
        Reference total column count used to compute the default figure width when
        `fixed_figsize` is None. Default 16 keeps legacy sizing near the original layout.
    col_ratio_current, col_ratio_time_total, col_ratio_colorbar
        Column width ratios for gridspec. The sum of *time columns* is kept constant by
        distributing `col_ratio_time_total` across the `n_time_panels` columns.
    """

    # style / colors
    style: Dict[str, Any] = field(default_factory=lambda: SCIENCEPLOT)
    sfr_color: str = "inferno"
    mass_color: str = "jet"

    # ranges and bins
    r_max: Optional[float] = None

    # Lookback-time range for display + binning (Gyr)
    t_range: Tuple[float, float] = (0.0, 14.)

    # Cosmic time at z=0 used for conversion: lookback = age_max - tform
    age_max: float = 13.80272

    # Mass-panel time binning in lookback time
    n_time_panels: int = 14

    r_round_to: float = 5.0
    r_nbins: int = 90
    t_nbins: int = 420
    nbins_xy: int = 200

    # figure layout
    frac: float = 1.2
    dpi: int = 150
    wspace: float = 0.0
    hspace: float = 0.0

    # stable sizing / proportions across different n_time_panels
    fixed_figsize: Optional[Tuple[float, float]] = None
    base_n_col_for_size: int = 16
    col_ratio_current: float = 1.0
    col_ratio_time_total: Optional[float] = None
    col_ratio_colorbar: float = 1.0

    # normalization and color scale
    face_vmin_pct: float = 0.1
    face_vmax_pct: float = 99.95

class PlotHandles(TypedDict):
    axes: Dict[str, object]
    images: Dict[str, AxesImage]
    colorbars: Dict[str, Colorbar]

def _resolve_lookback_edges(options: SFHPlotConfig) -> np.ndarray:
    """Resolve lookback-time bin edges for mass panels.

    Parameters
    ----------
    options
        Plot configuration.

    Returns
    -------
    edges : ndarray, shape (n_edges,)
        Lookback-time bin edges (Gyr). Uniform edges spanning `options.t_range` with
        `options.n_time_panels` bins are returned.

    Raises
    ------
    ValueError
        If ranges, counts, or edges are invalid.
    """
    t0, t1 = map(float, options.t_range)
    if not (t1 > t0):
        raise ValueError(f"Invalid t_range={options.t_range}: require t_range[1] > t_range[0].")
    if t0 < 0:
        raise ValueError(f"Invalid t_range={options.t_range}: lookback time must be >= 0.")

    n = int(options.n_time_panels)
    if n <= 0:
        raise ValueError("options.n_time_panels must be a positive integer.")
    return np.linspace(t0, t1, n + 1)


def _robust_log_percentile_clim(im: np.ndarray, vmin_pct: float, vmax_pct: float) -> Tuple[float, float]:
    """Robust (vmin, vmax) for log-scaled images from percentiles of positive pixels."""
    pos = im[np.isfinite(im) & (im > 0)]
    if pos.size == 0:
        warning("No positive pixels found for log-scaled color limits; using default (1e-6, 1.0).")
        return 1e-6, 1.0
    vmax = float(np.percentile(pos, vmax_pct))
    vmin = float(np.percentile(pos, vmin_pct))
    if not np.isfinite(vmax) or vmax <= 0:
        warning("Invalid vmax for log-scaled color limits; using default 1.0.")
        vmax = 1.0
    if not np.isfinite(vmin) or vmin <= 0:
        warning("Invalid vmin for log-scaled color limits; using default 1e-6.")
        vmin = min(1e-6, vmax * 1e-6)
    if vmin >= vmax:
        warning("vmin >= vmax for log-scaled color limits; adjusting vmin.")
        vmin = max(1e-6, vmax * 1e-6)
    return vmin, vmax

def _hist_imshow(
    ax: Axes,
    snap: "SimSnap",
    xkey: str,
    ykey: str,
    *,
    weights_key: str = "mass",
    density: bool = False,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nbins: int,
    cmap: str,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Compute 2D hist then draw into `ax` with consistent styling."""
    im, _, _ = hist_2d(
        snap.s[xkey], snap.s[ykey],
        weights=snap.s[weights_key],
        density=density,
        x_range=x_range,
        y_range=y_range,
        nbins=nbins,
    )
    ax.imshow(
        im,
        origin="lower",
        extent=(*x_range, *y_range),
        norm="log",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return im

def _plot_xy_xz_column(
    mass_panels: NDArray[np.object_],
    row0: int,
    col: int,
    snap: "SimSnap",
    *,
    density: bool,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    nbins: int,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Plot XY into (row0, col) and XZ into (row0+1, col)."""
    _hist_imshow(
        mass_panels[row0, col], snap, "x", "y",
        density=density, x_range=x_range, y_range=y_range,
        nbins=nbins, cmap=cmap, vmin=vmin, vmax=vmax,
    )
    _hist_imshow(
        mass_panels[row0 + 1, col], snap, "x", "z",
        density=density, x_range=x_range, y_range=y_range,
        nbins=nbins, cmap=cmap, vmin=vmin, vmax=vmax,
    )


def _select_current_by_lookback(current: "SimSnap", lb_min: float, lb_max: float) -> "SimSnap":
    sel = (current.s["age"] > lb_min) & (current.s["age"] < lb_max)
    return current.s[sel]


def _select_birth_by_tform(snap: "SimSnap", tf_min: float, tf_max: float) -> "SimSnap":
    sel = (snap.s["tform"] > tf_min) & (snap.s["tform"] < tf_max)
    return snap.s[sel]

def plot_sfr_evolution(
    current: "SimSnap", 
    birth_centered: "SimSnap", 
    birth_aligned: "SimSnap",
    sfh_color: str = "inferno",
    mass_color: str = "jet",
    r_max: Optional[float] = None,
    t_range: Tuple[float, float] = (0.0, 14.),
    options: Optional[SFHPlotConfig] = None,
    return_handles: bool = False,
    **kwargs: Any,
    ) -> Union[Figure, Tuple[Figure, PlotHandles]]:
    """
    Plot SFR radial evolution + mass panels.

    Time conventions
    ----------------
    - SFH axis uses lookback time (Gyr): lookback = options.age_max - tform
    - options.t_range is the lookback-time range displayed and used for mass-panel binning.
    """
    if options is None:
        options = SFHPlotConfig(
            style = kwargs.get('style', SCIENCEPLOT),
            sfr_color = sfh_color,
            mass_color = mass_color,
            t_range=t_range,
            r_max = r_max,
            **kwargs
        )

    # ---- basic option validation (fail fast) ----
    if options.r_round_to <= 0:
        raise ValueError("options.r_round_to must be > 0.")
    if options.r_nbins <= 0 or options.t_nbins <= 0:
        raise ValueError("options.r_nbins and options.t_nbins must be positive integers.")
    if int(options.nbins_xy) <= 0:
        raise ValueError("options.nbins_xy must be a positive integer.")
    if not (0.0 <= options.face_vmin_pct < options.face_vmax_pct <= 100.0):
        raise ValueError("face_vmin_pct/face_vmax_pct must satisfy 0<=vmin<vmax<=100.")
    if options.dpi <= 0:
        raise ValueError("options.dpi must be > 0.")

    plt.rcParams.update(options.style if options.style is not None else SCIENCEPLOT)

    # -------- ranges --------
    if options.r_max is None:
        r95 = float(np.percentile(current.s['r'],95)) # include 95% particles
    else:
        r95 = float(options.r_max)
        if r95 <= 0:
            raise ValueError("options.r_max must be > 0.")
        

    region_r = float((r95 // options.r_round_to + 1) * options.r_round_to)
    r_range = (0.0, region_r)
    

    # lookback-time range for display/bins
    t_range = (float(options.t_range[0]), float(options.t_range[1]))
    age_max = float(options.age_max)

    # -------- mass-panel time bins (lookback) + corresponding tform bins --------
    lb_edges = _resolve_lookback_edges(options)         # lookback edges within t_range
    lb_bins_min = lb_edges[:-1]
    lb_bins_max = lb_edges[1:]

    # convert lookback bin -> tform bin for birth_* selections
    # tform = age_max - lookback
    tform_bins_min = age_max - lb_bins_max
    tform_bins_max = age_max - lb_bins_min


    # -------- fig grid (dynamic columns, stable figsize) --------
    n_row = 9
    n_time_panels = int(lb_edges.size - 1)
    n_col = 1 + n_time_panels + 1  # [Current] + [time panels] + [colorbar]
    
    if options.fixed_figsize is not None:
        figsize = (float(options.fixed_figsize[0]), float(options.fixed_figsize[1]))
    else:
        # keep overall figure size near legacy default (16 columns wide)
        figsize = (float(options.frac) * float(options.base_n_col_for_size), float(options.frac) * float(n_row))

    # keep SFH/time area width roughly constant across different n_time_panels
    time_total = (float(options.col_ratio_time_total) 
                if options.col_ratio_time_total is not None 
                else float(options.base_n_col_for_size - 2) )
    time_each = (time_total / float(n_time_panels) 
                if n_time_panels > 0 else time_total)

    width_ratios = ([float(options.col_ratio_current)] 
                    + [time_each] * n_time_panels 
                    + [float(options.col_ratio_colorbar)])


    fig = plt.figure(dpi=int(options.dpi), figsize=figsize)
    gs = fig.add_gridspec(
        n_row, n_col, 
        wspace=float(options.wspace), 
        hspace=float(options.hspace), 
        width_ratios=width_ratios)

    handles: PlotHandles = {"axes":{}, "images":{}, "colorbars":{}}

    not_use = plt.subplot(gs[:3,0]) 
    not_use.set_axis_off()

    sfh_ax = plt.subplot(gs[:3,1:-1])
    handles['axes']['sfh'] = sfh_ax

    # SFH map expects tform + t_max=age_max, and uses lookback t_range for display
    im = sfr_virus_radial_evolution(
        birth_aligned['tform'], birth_aligned['mass'], birth_aligned['r'], 
        r_range=r_range, t_range=t_range, r_nbins=options.r_nbins, t_nbins=options.t_nbins, t_max=age_max
    )

    sfh_im = sfh_ax.imshow(
        im, origin='lower', extent=(*t_range,*r_range), aspect='auto', norm='log',
        cmap=options.sfr_color
    )
    handles['images']['sfh'] = sfh_im

    sfh_ax_facecolor = sfh_im.cmap(sfh_im.norm(sfh_im.get_clim()[0]))
    sfh_ax.set_facecolor(sfh_ax_facecolor)
    sfh_ax.tick_params(axis='both',which='both',direction='out')
    sfh_ax.xaxis.set_ticks_position('top')
    sfh_ax.xaxis.set_label_position('top')

    # IMPORTANT: xlim follows user t_range (lookback)
    sfh_ax.set_xlim(*t_range)

    # (optional) keep the vertical marker at the right boundary of the shown range
    if t_range[0] < options.age_max < t_range[1]:
        sfh_ax.axvline(options.age_max, color='r', linewidth=1)

    sfh_ax.set_xlabel('Lookback time [Gyr]')
    sfh_ax.set_ylabel('R [kpc]')

    sfh_bar = plt.subplot(gs[:3, -1])
    cb_sfh = fig.colorbar(
        sfh_im,
        shrink=3/4,
        cax=sfh_bar,
        label=r"$\Sigma_{\rm SFR}\ [\rm M_\odot\ Gyr^{-1} \ kpc^{-2}]$",
        extend='both'
    )
    handles['colorbars']['sfh'] = cb_sfh
    sfh_bar_pos = sfh_bar.get_position()
    sfh_bar.set_position((sfh_bar_pos.x0, sfh_bar_pos.y0, sfh_bar_pos.width*0.1, sfh_bar_pos.height))
    sfh_bar.tick_params(direction='in', right=True)

    # mass panels
    mass_panels = np.array([[plt.subplot(gs[i+3,j]) for j in range(n_col-1)] for i in range(6)])
    handles['axes']['mass_panels'] = mass_panels

    x_range = (-region_r, region_r)
    y_range = (-region_r, region_r)
    nbins = int(options.nbins_xy)

    # --- Reference scaling from current XY in column 0 ---
    im_cur, _, _ = hist_2d(
        current.s["x"], current.s["y"],
        weights=current.s["mass"],
        x_range=x_range, y_range=y_range, nbins=nbins
    )
    face_vmin, face_vmax = _robust_log_percentile_clim(im_cur, options.face_vmin_pct, options.face_vmax_pct)

    # draw current col0 (and keep a handle for colorbar)
    mass_im = mass_panels[0, 0].imshow(
        im_cur,
        origin="lower",
        extent=(*x_range, *y_range),
        norm="log",
        cmap=options.mass_color,
        vmin=face_vmin,
        vmax=face_vmax,
    )

    # current XZ col0
    _hist_imshow(
        mass_panels[1, 0], current, "x", "z",
        density=False, x_range=x_range, y_range=y_range,
        nbins=nbins, cmap=options.mass_color, vmin=face_vmin, vmax=face_vmax
    )

    # --- Define three blocks: (row0, snapshot, density, selector) ---
    # rows: current(0,1), birth_aligned(2,3), birth_centered(4,5)
    blocks: List[Tuple[int, "SimSnap", bool, str]] = [
        (0, current, True, "current"),
        (2, birth_aligned, True, "birth"),
        (4, birth_centered, True, "birth"),
    ]

    # columns 1..n_time_panels correspond to bins k=0..n_time_panels-1
    for col in range(1, n_col - 1):
        k = col - 1
        if k >= len(lb_bins_min):
            break

        lb_min = float(lb_bins_min[k])
        lb_max = float(lb_bins_max[k])
        tf_min = float(tform_bins_min[k])
        tf_max = float(tform_bins_max[k])

        for row0, snap, density, mode in blocks:
            if mode == "current":
                sub = _select_current_by_lookback(current, lb_min, lb_max)
                # `sub` is a SimSnap view (already .s-like); wrap expected interface
                # hist_2d usage below assumes `sub` has `.s[...]`, so use `current.s[sel]` result directly:
                # Here `_select_current_by_lookback` returns `current.s[sel]`, which is a SimSnap view.
                snap_view = sub
            else:
                snap_view = _select_birth_by_tform(snap, tf_min, tf_max)

            _plot_xy_xz_column(
                mass_panels, row0, col, snap_view,
                density=density,
                x_range=x_range, y_range=y_range,
                nbins=nbins,
                cmap=options.mass_color,
                vmin=face_vmin, vmax=face_vmax,
            )

    # --- Birth blocks col0 are full snapshot (already drawn for current above) ---
    _plot_xy_xz_column(
        mass_panels, 2, 0, birth_aligned,
        density=True, x_range=x_range, y_range=y_range,
        nbins=nbins, cmap=options.mass_color, vmin=face_vmin, vmax=face_vmax
    )
    _plot_xy_xz_column(
        mass_panels, 4, 0, birth_centered,
        density=True, x_range=x_range, y_range=y_range,
        nbins=nbins, cmap=options.mass_color, vmin=face_vmin, vmax=face_vmax
    )

    mass_panels_face_color = mass_im.cmap(mass_im.norm(mass_im.get_clim()[0]))
    for i in mass_panels.flatten():
        i.set_facecolor(mass_panels_face_color)
        
    for i in range(mass_panels.shape[0]):
        for j in range(mass_panels.shape[1]):
            if (i!=mass_panels.shape[0]-1) or (j%2 == 1):
                    mass_panels[i,j].set_xticklabels([])
            if j!=0:
                mass_panels[i,j].set_yticklabels([])
            mass_panels[i,j].tick_params(axis='both',which='both',direction='out')
            if i==0:
                mass_panels[i,j].tick_params(axis='x',which='both',direction='in')
            if j==mass_panels.shape[1]-1:
                mass_panels[i,j].tick_params(axis='y',which='both',direction='in')


    pos0 = mass_panels[0,0].get_position()
    pos1 = mass_panels[1,0].get_position()
    ycenter = (pos0.y0 + pos1.y1) / 2
    xleft = pos0.x0
    fig.text(xleft - 0.03, ycenter, 'Current', va='center', ha='right', rotation='vertical')


    pos0 = mass_panels[2,0].get_position()
    pos1 = mass_panels[3,0].get_position()
    ycenter = (pos0.y0 + pos1.y1) / 2
    xleft = pos0.x0
    fig.text(xleft - 0.03, ycenter, 'Birth Aligned', va='center', ha='right', rotation='vertical')

    pos0 = mass_panels[4,0].get_position()
    pos1 = mass_panels[5,0].get_position()
    ycenter = (pos0.y0 + pos1.y1) / 2
    xleft = pos0.x0
    fig.text(xleft - 0.03, ycenter, 'Birth Centered', va='center', ha='right', rotation='vertical')

    pos0 = mass_panels[-1,0].get_position()
    pos1 = mass_panels[-1,-1].get_position()
    xcenter = (pos0.x0 + pos1.x1) / 2
    yleft = pos0.y0
    fig.text(xcenter, yleft - 0.03, 'X [kpc]', va='top', ha='center')


    mass_bar = plt.subplot(gs[3:, -1])
    cb_mass = fig.colorbar(
        mass_im,
        shrink=3/4,
        cax=mass_bar,
        label=r"$\Sigma_* \ [\rm M_\odot/kpc^2]$",
        extend='both'
    )
    handles['colorbars']['mass'] = cb_mass
    mass_bar_bar_pos = mass_bar.get_position()
    mass_bar.set_position(
        (mass_bar_bar_pos.x0, mass_bar_bar_pos.y0, mass_bar_bar_pos.width*0.1, mass_bar_bar_pos.height)
        )
    mass_bar.tick_params(direction='in', right=True)
    
    if return_handles:
        return fig, handles
    return fig
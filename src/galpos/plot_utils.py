
from typing import Tuple, Optional

import numpy as np

def hist_2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    parameters: Optional[np.ndarray] = None,
    density: bool = True,
    gridsize: Tuple[int, int] = (100, 100),
    nbins: Optional[int] = None,
    x_logscale: bool = False,
    y_logscale: bool = False,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D histogram from input data.

    Parameters
    ----------
    x : array-like
        Input data for the x-axis.
    y : array-like
        Input data for the y-axis.
    weights : array-like, optional
        Weights for each data point. Default is None.
    parameters : array-like, optional
        Additional parameters for weighted histograms. Default is None.
    density : bool, optional
        If True, normalize the histogram to form a probability density. Default is True.
    gridsize : tuple of int, optional
        Number of bins for the histogram in (y, x) directions. Default is (100, 100).
    nbins : int, optional
        Number of bins for both axes (overrides gridsize if provided). Default is None.
    x_logscale : bool, optional
        If True, apply log scaling to the x-axis. Default is False.
    y_logscale : bool, optional
        If True, apply log scaling to the y-axis. Default is False.
    x_range : list or tuple, optional
        Range for the x-axis as [min, max]. Default is None.
    y_range : list or tuple, optional
        Range for the y-axis as [min, max]. Default is None.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    hist : 2D ndarray
        The 2D histogram array.
    xs : ndarray
        Bin centers for the x-axis.
    ys : ndarray
        Bin centers for the y-axis.

    Examples
    --------
    >>> im,xs,ys = hist_2d(x, y, weights=weights, parameters=parameters,x_range=x_range,y_range=y_range)
    >>> plt.imshow(im, origin='lower', extent=(*x_range,*y_range))
    """
    if nbins is not None:
        gridsize = (nbins, nbins)

    if y_range is not None:
        if len(y_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        y_range = (
            (np.log10(np.min(y)), np.log10(np.max(y)))
            if y_logscale
            else (np.min(y), np.max(y))
        )

    if x_range is not None:
        if len(x_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        x_range = (
            (np.log10(np.min(x)), np.log10(np.max(x)))
            if x_logscale
            else (np.min(x), np.max(x))
        )

    x = np.log10(x) if x_logscale else x
    y = np.log10(y) if y_logscale else y

    ind = np.where(
        (x > x_range[0]) & (x < x_range[1]) & (y > y_range[0]) & (y < y_range[1])
    )

    x = x[ind[0]]
    y = y[ind[0]]

    if weights is not None:
        weights = weights[ind[0]]

    if parameters is not None:
        parameters = parameters[ind[0]]
        if weights is None:
            weights = np.ones_like(parameters)

    def _histogram_generator(weights):
        return np.histogram2d(
            y, x, weights=weights, bins=[gridsize[1], gridsize[0]], range=[y_range, x_range]
        )

    if parameters is not None:
        hist, ys, xs = _histogram_generator(weights * parameters)
        hist_norm, _, _ = _histogram_generator(weights)
        valid = hist_norm > 0
        hist[valid] /= hist_norm[valid]
    else:
        hist, ys, xs = _histogram_generator(weights)

    if density:
        area = np.diff(xs).reshape(-1, 1) * np.diff(ys)
        hist = hist / area

    xs = 0.5 * (xs[:-1] + xs[1:])
    ys = 0.5 * (ys[:-1] + ys[1:])

    return hist,xs,ys

def sfr_virus_radial_evolution(
    tform: np.ndarray, 
    mass: np.ndarray, 
    r: np.ndarray, 
    r_range: tuple[float, float] = (0,30.), 
    t_range: tuple[float, float] = (0, 13.80272), 
    r_nbins: int =100, 
    t_nbins: int = 200
    ) -> np.ndarray:
    """ 
    Compute the star formation rate (SFR) in a 2D histogram of time and radius.

    Parameters
    ----------
    tform: array-like
        The formation time of the stars.
    mass: array-like
        The mass of the stars.
    r: array-like
        The radius of the stars. (birth positon)
    r_range: tuple
        The range of radii to consider.
    t_range: tuple
        The range of formation times to consider.
    r_nbins: int
        The number of bins in the radius dimension.
    t_nbins: int
        The number of bins in the time dimension.

    Returns
    -------
    im: np.ndarray
        The 2D histogram of star formation rates density.

    Examples
    --------
    >>> im = sfr_virus_radial_evolution(tform, mass, r)
    >>> plt.imshow(im, origin='lower', extent=(*t_range,*r_range))
    
    """

    im, t_c, r_c = hist_2d(
        t_range[1]-tform, r, weights=mass, density=False,
        x_range=t_range,y_range=r_range,gridsize=(t_nbins,r_nbins))

    delta_time = ((t_range[1]-t_range[0]) / t_nbins) # in Gyr
    im = im / delta_time

    delta_r = (r_range[1]-r_range[0]) / r_nbins
    area = 4*np.pi*(r_c+delta_r/2)**2-4*np.pi*(r_c-delta_r/2)**2

    im = im/area.reshape((im.shape[0],)+(1,)*(im.ndim-1))
    return im


def gaussian_filter(arr: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)


SCIENCEPLOT = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "xtick.direction": "in",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
}
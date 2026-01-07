"""
Trajectory utilities (translation only).

This module provides:

- helpers for periodic boxes (:func:`wrap_to_box`, :func:`unwrap_positions`)
- :class:`Trajectory`: continuous interpolation of position/velocity/(optional) acceleration

Interpolation methods
---------------------
Let sampled times be ``t_i`` and positions be ``x_i`` (vector in R^ndim). The goal is to
construct a continuous function ``x(t)`` and its derivatives ``v(t)=dx/dt`` and optionally
``a(t)=d^2x/dt^2``.

Supported strategies:

1) ``method='spline'`` (Cubic Hermite spline)
   You provide both ``x_i`` and ``v_i``. On each interval ``[t_i, t_{i+1}]``, a cubic
   polynomial is constructed so that:

   - ``x(t_i) = x_i``, ``x(t_{i+1}) = x_{i+1}``
   - ``x'(t_i) = v_i``, ``x'(t_{i+1}) = v_{i+1}``

   This yields a globally C^1 trajectory (continuous position and velocity).

2) ``method='polynomial'`` (piecewise cubic/quintic with endpoint constraints)
   - Cubic: match position + velocity at endpoints (same constraints as Hermite).
   - Quintic: match position + velocity + acceleration at endpoints:

     ``x(t_i)=x_i, x'(t_i)=v_i, x''(t_i)=a_i`` and same at ``t_{i+1}``.

   This yields C^2 continuity *inside each interval* and a physically smoother profile when
   accelerations are meaningful and not too noisy.

3) ``method='pchip'`` (PCHIP on positions)
   You provide only ``x_i``. PCHIP builds a shape-preserving, piecewise cubic interpolant
   for each component, and velocities/accelerations are derived by differentiating the
   interpolant. This is useful when you do not trust provided velocities.

Periodic boundary conditions
----------------------------
If a periodic box of size ``L`` is used, positions may jump by ~L at wrap boundaries.
We “unwrap” the samples to a continuous representation using the minimal-image convention:

``Δ = x_i - x_{i-1}``
``Δ ← Δ - round(Δ / L) * L``
``x_unwrapped[i] = x_unwrapped[i-1] + Δ``

Evaluation can optionally wrap back via ``x mod L``.

When to use which method
------------------------
- Use **'spline'** when you have trustworthy velocities and want a smooth C^1 path.
  Typical for simulation outputs storing consistent pos/vel.
- Use **'polynomial'** with accelerations when you want smoother kinematics (C^2-like)
  and accelerations are not dominated by noise.
- Use **'pchip'** when velocities are missing/unreliable, and you prefer shape-preserving
  interpolation over potentially oscillatory splines.

Examples
--------
Basic usage with explicit velocity samples:

>>> import numpy as np
>>> from galpos.orbits import Trajectory
>>> t = np.array([0., 1., 2.])
>>> pos = np.array([[0., 0., 0.],
...                 [1., 0., 0.],
...                 [2., 0., 0.]])
>>> vel = np.array([[1., 0., 0.],
...                 [1., 0., 0.],
...                 [1., 0., 0.]])
>>> tr = Trajectory(t, pos, vel, method="spline")
>>> tr(0.5)[0]
array([0.5, 0. , 0. ])

Periodic box: unwrap internally, wrap on output:

>>> L = 10.0
>>> pos = np.array([[9.5, 0, 0],
...                 [0.2, 0, 0],
...                 [1.0, 0, 0]], dtype=float)
>>> vel = np.array([[1, 0, 0],
...                 [1, 0, 0],
...                 [1, 0, 0]], dtype=float)
>>> tr = Trajectory(t, pos, vel, box_size=L, method="spline")
>>> p_unwrapped, _ = tr(0.5, wrap=False)
>>> p_wrapped, _ = tr(0.5, wrap=True)
>>> p_unwrapped[0] > 9
True
>>> 0 <= p_wrapped[0] < L
True
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator, PPoly
from scipy.linalg import solve


def wrap_to_box(x: ArrayLike, L: float) -> np.ndarray:
    """
    Wrap positions to periodic box.
    
    Parameters
    ----------
    x : array_like
        Positions to wrap
    L : float
        Box size
        
    Returns
    -------
    np.ndarray
        Wrapped positions in range [0, L)
    """
    return np.mod(x, L)

def unwrap_positions(pos: ArrayLike, L: float) -> np.ndarray:
    """
    Unwrap periodic boundary jumps to create continuous trajectories.

    Mathematical idea
    -----------------
    Given wrapped samples x_i in [0, L), we reconstruct a continuous series by applying
    a minimal-image correction to each step:

    ``Δ_i = x_i - x_{i-1}``
    ``Δ_i ← Δ_i - round(Δ_i / L) * L``
    ``x_unwrapped[i] = x_unwrapped[i-1] + Δ_i``

    Parameters
    ----------
    pos : array_like
        Array of positions with possible boundary jumps.
    L : float
        Box size.

    Returns
    -------
    np.ndarray
        Unwrapped positions with continuous trajectories.
    """
    pos = np.asarray(pos, dtype=float)
    unwrapped = pos.copy()
    for i in range(1, len(pos)):
        delta = pos[i] - pos[i-1]
        delta = delta - np.round(delta / L) * L
        unwrapped[i] = unwrapped[i-1] + delta
    return unwrapped

class PolynomialInterpolator(PPoly):
    """
    Piecewise polynomial interpolation with endpoint derivative constraints.

    Mathematical form
    -----------------
    On each interval [x_i, x_{i+1}] we use a local coordinate τ = (t - x_i) with
    τ in [0, dx], dx = x_{i+1}-x_i, and fit:

    - Cubic:   p(τ)=a0 + a1 τ + a2 τ^2 + a3 τ^3
      constraints: p(0)=y_i, p(dx)=y_{i+1}, p'(0)=y'_i, p'(dx)=y'_{i+1}

    - Quintic: p(τ)=a0 + a1 τ + a2 τ^2 + a3 τ^3 + a4 τ^4 + a5 τ^5
      constraints additionally include p''(0)=y''_i and p''(dx)=y''_{i+1}

    The implementation solves the resulting linear system per segment (and per component).

    When to use
    -----------
    - Cubic is appropriate when you trust velocities but do not have accelerations.
    - Quintic is appropriate when you also have accelerations and want smoother motion,
      but it can overfit if accelerations are noisy.
    """

    def __init__(self, 
                x: ArrayLike, 
                y: ArrayLike, 
                dydx: ArrayLike, 
                d2ydx2: Optional[ArrayLike] = None, 
                extrapolate: bool = False, 
                axis: int = 0):
        """
        Initialize the polynomial interpolator.
        
        Parameters
        ----------
        x : array_like
            1-D array of increasing sample points
        y : array_like
            Array of function values at sample points
        dydx : array_like
            Array of first derivatives at sample points
        d2ydx2 : array_like, optional
            Array of second derivatives at sample points.
            If provided, a quintic polynomial is used; otherwise cubic.
        extrapolate : bool, default=False
            Whether to extrapolate beyond the given range
        axis : int, default=0
            Axis along which y is assumed to be varying
        """
        x = np.asarray(x)
        y = np.asarray(y)
        dydx = np.asarray(dydx)

        self.axis = axis
        
        # Handle multidimensional inputs by moving the interpolation axis to the front
        if axis != 0:
            y = np.moveaxis(y, axis, 0)
            dydx = np.moveaxis(dydx, axis, 0)
            if d2ydx2 is not None:
                d2ydx2 = np.moveaxis(d2ydx2, axis, 0)
        
        is_quintic = d2ydx2 is not None
        self.y_values = y
        self.dydx_values = dydx
        
        # Process second derivatives if provided
        if is_quintic:
            self.d2ydx2_values = np.asarray(d2ydx2)
            k = 6  # quintic: order = 5, k = order + 1
        else:
            self.d2ydx2_values = None
            k = 4  # cubic: order = 3, k = order + 1
        
        # Reshape y and derivatives for vectorized computation if multi-dimensional
        orig_shape = y.shape
        ndim = len(orig_shape)
        
        if ndim > 1:
            # Reshape to (n_points, n_values) for vectorized computation
            n_points = orig_shape[0]  
            n_values = np.prod(orig_shape[1:])
            y_reshaped = y.reshape(n_points, n_values)
            dydx_reshaped = dydx.reshape(n_points, n_values)
            
            if is_quintic:
                d2ydx2_reshaped = self.d2ydx2_values.reshape(n_points, n_values)
            
            # Process each dimension separately and stack results
            c_list = []
            for j in range(n_values):
                y_j = y_reshaped[:, j]
                dydx_j = dydx_reshaped[:, j]
                d2ydx2_j = None if not is_quintic else d2ydx2_reshaped[:, j]
                
                c_j = self._compute_coefficients(x, y_j, dydx_j, d2ydx2_j, k)
                c_list.append(c_j)
            
            # Stack coefficients along a new dimension
            c = np.stack(c_list, axis=-1)
            # Reshape to original dimensions
            c = c.reshape(c.shape[0], c.shape[1], *orig_shape[1:])
            
        else:
            # 1D case - compute coefficients directly
            c = self._compute_coefficients(x, y, dydx, d2ydx2, k)
        
        # Initialize the base class with coefficients in PPoly format
        super().__init__(c, x, extrapolate)
        
    def _compute_coefficients(self, 
                             x: np.ndarray, 
                             y: np.ndarray, 
                             dydx: np.ndarray, 
                             d2ydx2: Optional[np.ndarray], 
                             k: int) -> np.ndarray:
        """
        Compute polynomial coefficients for a single set of values.
        
        Parameters
        ----------
        x : np.ndarray
            1-D array of sample points
        y : np.ndarray
            Function values at sample points
        dydx : np.ndarray
            First derivatives at sample points
        d2ydx2 : np.ndarray or None
            Second derivatives at sample points, or None
        k : int
            Order of polynomial + 1 (4 for cubic, 6 for quintic)
            
        Returns
        -------
        np.ndarray
            Array of polynomial coefficients
        
        Raises
        ------
        ValueError
            If input arrays have inconsistent lengths
        """
        # Check input array lengths
        n = len(x)
        if len(y) != n or len(dydx) != n:
            raise ValueError(f"Input arrays must have the same length. Got: "
                            f"x:{len(x)}, y:{len(y)}, dydx:{len(dydx)}")
        
        c = np.zeros((k, len(x) - 1), dtype=float)
        
    
        if d2ydx2 is not None and d2ydx2.shape[0] != n:
            raise ValueError(f"d2ydx2 length {len(d2ydx2)} does not match x length {n}")
        
        for i in range(len(x) - 1):
            x1, x2 = x[i], x[i + 1]
            y1, y2 = y[i], y[i + 1]
            dy1, dy2 = dydx[i], dydx[i + 1]
            
            # Use local coordinates to build equation system
            dx = x2 - x1
            
            if d2ydx2 is not None:
                d2y1, d2y2 = d2ydx2[i], d2ydx2[i + 1]
                
                A = np.array([
                    [1, 0, 0, 0, 0, 0],           # p(0) = y1
                    [1, dx, dx**2, dx**3, dx**4, dx**5],   # p(dx) = y2
                    [0, 1, 0, 0, 0, 0],           # p'(0) = dy1
                    [0, 1, 2*dx, 3*dx**2, 4*dx**3, 5*dx**4], # p'(dx) = dy2
                    [0, 0, 2, 0, 0, 0],           # p''(0) = d2y1
                    [0, 0, 2, 6*dx, 12*dx**2, 20*dx**3],   # p''(dx) = d2y2
                ])
                
                b = np.array([y1, y2, dy1, dy2, d2y1, d2y2])
                coeffs = solve(A, b)
                
            else:
                A = np.array([
                    [1, 0, 0, 0],           # p(0) = y1
                    [1, dx, dx**2, dx**3],   # p(dx) = y2
                    [0, 1, 0, 0],           # p'(0) = dy1
                    [0, 1, 2*dx, 3*dx**2],  # p'(dx) = dy2
                ])
                
                b = np.array([y1, y2, dy1, dy2])
                coeffs = solve(A, b)
            
            # Store coefficients: highest order first
            c[:, i] = coeffs[::-1]
            
        return c

    @classmethod
    def construct_fast(cls, 
                      c: np.ndarray, 
                      x: np.ndarray, 
                      extrapolate: Optional[bool] = None, 
                      axis: int = 0) -> 'PolynomialInterpolator':
        """
        Construct the piecewise polynomial without validation checks.

        Parameters
        ----------
        c : ndarray
            Array of polynomial coefficients for each segment
        x : ndarray
            1-D array of sample points
        extrapolate : bool, optional
            Whether to extrapolate beyond the given range
        axis : int, default=0
            Axis along which y is assumed to vary
            
        Returns
        -------
        PolynomialInterpolator
            A new interpolator instance
        """
        self = super().construct_fast(c, x, extrapolate, axis)
        self.axis = axis
        return self

    def derivative(self, nu: int = 1)-> 'PolynomialInterpolator':
        """
        Return a piecewise polynomial representing the derivative.
        
        Parameters
        ----------
        nu : int, default=1
            Order of derivative
            
        Returns
        -------
        PolynomialInterpolator
            Piecewise polynomial representing the derivative
        """
        ppoly_deriv = super().derivative(nu)
        return self.construct_fast(
            ppoly_deriv.c, 
            ppoly_deriv.x, 
            ppoly_deriv.extrapolate, 
            self.axis
        )


class Trajectory:
    """
    Continuous trajectory model for translation (position/velocity/acceleration).

    Interpolation model (math)
    --------------------------
    Given sampled times t_i and positions x_i, the class constructs an interpolant x(t).
    Velocity and acceleration are obtained by analytic differentiation of the interpolant:

    - v(t) = d x(t) / dt
    - a(t) = d^2 x(t) / dt^2

    Method selection and assumptions
    --------------------------------
    - 'spline' (CubicHermiteSpline): assumes provided velocities v_i are consistent with x_i.
      Produces C^1 continuity globally.
    - 'polynomial' (PolynomialInterpolator):
      * cubic if only (x_i, v_i) are given
      * quintic if (x_i, v_i, a_i) are given
    - 'pchip' (PchipInterpolator): assumes only x_i are reliable; v(t), a(t) are derived
      from the fitted curve.

    Practical guidance
    ------------------
    - If you care about **physical kinematics**, prefer methods that use real velocities
      (Hermite/polynomial) rather than derivatives estimated from positions.
    - If positions are in a periodic box, set `box_size` so interpolation happens in an
      unwrapped space; otherwise you may interpolate across discontinuities.

    Parameters
    ----------
    times : array_like, shape (N,)
        Sample times (must be strictly increasing; unsorted inputs are sorted).
    positions : array_like, shape (N,) or (N, ndim)
        Sampled positions.
    velocities : array_like, optional
        Sampled velocities. Required for ``method='spline'`` and recommended for
        best physical fidelity.
    accelerations : array_like, optional
        Sampled accelerations. If provided, the implementation may use a quintic
        polynomial to satisfy endpoint constraints.
    box_size : float, optional
        Periodic box size. If set, positions are internally unwrapped for smooth
        interpolation. Use ``wrap=True`` at evaluation time to return wrapped positions.
    method : {'spline', 'polynomial', 'pchip'}, default='spline'
        Interpolation method.

    Notes
    -----
    - For scalar ``t``, :meth:`__call__` returns vectors with shape ``(ndim,)``.
      For array ``t``, it returns arrays of shape ``(M, ndim)``.
    - When `velocities` is omitted, derivatives may be estimated from positions.

    Examples
    --------
    See module-level examples above.
    """
    
    def __init__(self, 
                times: ArrayLike, 
                positions: ArrayLike, 
                velocities: Optional[ArrayLike] = None, 
                accelerations: Optional[ArrayLike] = None, 
                box_size: Optional[float] = None, 
                method: str = 'spline'):
        """
        Initialize a trajectory from positions, velocities and optional accelerations.
        
        Parameters
        ----------
        times : array_like
            Time array with shape (N,)
        positions : array_like
            Position array with shape (N,) or (N, ndim)
        velocities : array_like, optional
            Velocity array with shape (N,) or (N, ndim)
        accelerations : array_like, optional
            Acceleration array with shape (N,) or (N, ndim)
        box_size : float, optional
            Size of the periodic box for wrapping positions
        method : {'spline', 'polynomial', 'pchip'}, default='spline'
            Interpolation method:
            - 'spline': CubicHermiteSpline (requires velocities)
            - 'polynomial': PolynomialInterpolator (cubic or quintic)
            - 'pchip': PchipInterpolator (can estimate velocities)
        """
        # Ensure inputs are numpy arrays
        times = np.asarray(times)
        positions = np.asarray(positions)
        
        # Sort by time if needed
        if not np.all(np.diff(times) > 0):
            idx = np.argsort(times)
            times = times[idx]
            positions = positions[idx]
            if velocities is not None:
                velocities = np.asarray(velocities)[idx]
            if accelerations is not None:
                accelerations = np.asarray(accelerations)[idx]

        assert any(np.diff(times) > 0), "Times must be strictly increasing"
        
        self.times: np.ndarray = times
        self.box_size: Optional[float] = box_size
        self.method: str = method
        
        self.positions: np.ndarray = positions
        
        # Determine dimensionality of the trajectory
        if positions.ndim == 1:
            self.ndim = 1
        else:
            self.ndim = positions.shape[1]
        
        self.velocities: np.ndarray
        self.accelerations: np.ndarray
        self.unwrapped_positions: Optional[np.ndarray] = None
        self.spline: Union[
            PolynomialInterpolator, 
            CubicHermiteSpline, 
            PchipInterpolator
        ]
        
        
        # Handle periodic boundary conditions
        if box_size is not None:
            self.unwrapped_positions = unwrap_positions(positions, box_size)
            pos = self.unwrapped_positions
        else:
            pos = self.positions
            
            
        # When accelerations are provided, use quintic polynomial interpolation
        if accelerations is not None:
            if method != 'polynomial':
                print("Warning: accelerations provided, "
                      "switching to 'polynomial' method")
                method = 'polynomial'

            self.accelerations = np.asarray(accelerations)
            if velocities is None:
                # Estimate velocities if not provided
                self.velocities = np.zeros_like(pos)
                dt = np.diff(times)
                dp = np.diff(pos, axis=0)
                self.velocities[:-1] = dp / dt[:, np.newaxis]
                self.velocities[-1] = self.velocities[-2]  # Copy last velocity
            else:
                self.velocities = np.asarray(velocities)
                
            # Use PolynomialInterpolator with axis=1 for multi-dimensional data
            self.spline = PolynomialInterpolator(
                times, pos, self.velocities, self.accelerations,
            )

         # When velocities and accelerations are not provided, we need to estimate them
        elif velocities is None:
            # Use PchipInterpolator which doesn't require explicit derivatives
            if method != 'pchip':
                print("Warning: velocities and accelerations not provided, "
                      f"switching to 'pchip' method instead of '{method}'")
                method = 'pchip'

            self.spline = PchipInterpolator(times, pos)
            # Calculate velocities from the interpolator derivatives
            self.velocities = self.spline.derivative()(times)
            self.accelerations = self.spline.derivative(2)(times)

        else:
            # We have explicit velocities
            self.velocities = np.asarray(velocities)
            
            if method == 'spline':
                self.spline = CubicHermiteSpline(times, pos, self.velocities)
            elif method == 'polynomial':
                self.spline = PolynomialInterpolator(times, pos, self.velocities)
            elif method == 'pchip':
                self.spline = PchipInterpolator(times, pos)
                # Override provided velocities with those from the PCHIP interpolator
                self.velocities = self.spline.derivative()(times)
            self.accelerations = self.spline.derivative(2)(times)

    def __call__(self, 
                 t: Union[float, ArrayLike], 
                 wrap: bool = False, 
                 extrapolate: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate position and velocity at time(s) ``t``.

        Parameters
        ----------
        t : float or array_like
            Query time(s).
        wrap : bool, default=False
            If True and `box_size` is set, wrap positions into ``[0, box_size)``.
        extrapolate : bool, default=False
            If True, allow evaluation outside the sampled time range.

        Returns
        -------
        position : ndarray
            Position(s).
        velocity : ndarray
            Velocity(s).

        Examples
        --------
        >>> import numpy as np
        >>> from galpos.orbits import Trajectory
        >>> t = np.array([0., 1., 2.])
        >>> pos = np.array([[0., 0., 0.],
        ...                 [1., 0., 0.],
        ...                 [2., 0., 0.]])
        >>> vel = np.ones_like(pos)
        >>> tr = Trajectory(t, pos, vel)
        >>> p, v = tr([0.5, 1.5])
        >>> p.shape, v.shape
        ((2, 3), (2, 3))
        """
        # Convert to array if scalar
        t_array = np.atleast_1d(t)
        scalar_input = np.isscalar(t)
        
        # Evaluate position and velocity using the interpolators
        pos = self.spline(t_array, extrapolate=extrapolate)
        vel = self.spline.derivative()(t_array, extrapolate=extrapolate)

        # Apply wrapping if needed
        if self.box_size is not None and wrap:
            pos = wrap_to_box(pos, self.box_size)
        
        # Return scalar or array depending on input
        if scalar_input:
            return pos[0], vel[0]
        return pos, vel

    def get_acceleration(self, 
                         t: Union[float, ArrayLike], 
                         extrapolate: bool = False
                         ) -> np.ndarray:
        """
        Evaluate acceleration at specified time(s).
        
        Parameters
        ----------
        t : float or array_like
            Time(s) at which to evaluate acceleration
        extrapolate : bool, default=False
            If True, extrapolate beyond the time range of data
            
        Returns
        -------
        np.ndarray
            Acceleration values with shape (ndim,) or (N, ndim)
        """
        t_array = np.atleast_1d(t)
        scalar_input = np.isscalar(t)
        
        acc = self.spline.derivative(nu=2)(t_array, extrapolate=extrapolate)
                        
        if scalar_input:
            return acc[0]
        return acc
        
    @classmethod
    def from_orbit(cls, 
                  pos: ArrayLike, 
                  vel: ArrayLike, 
                  t: ArrayLike, 
                  box_size: Optional[float] = None, 
                  method: str = 'polynomial') -> 'Trajectory':
        """
        Create a Trajectory from position, velocity, and time arrays.
        
        Parameters
        ----------
        pos : array_like
            Position array with shape (N,) or (N, ndim)
        vel : array_like
            Velocity array with shape (N,) or (N, ndim) 
        t : array_like
            Time array with shape (N,)
        box_size : float, optional
            Size of the periodic box
        method : str, default='polynomial'
            Interpolation method
            
        Returns
        -------
        Trajectory
            A new Trajectory object
        """
        return cls(
            times=t, positions=pos, velocities=vel, 
            box_size=box_size, method=method)
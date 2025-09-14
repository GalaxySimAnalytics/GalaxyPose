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
    
    Parameters
    ----------
    pos : array_like
        Array of positions with possible boundary jumps
    L : float
        Box size
        
    Returns
    -------
    np.ndarray
        Unwrapped positions with continuous trajectories
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
    Polynomial interpolation with position, velocity, and optional acceleration 
    constraints.
    
    Implements a piecewise polynomial interpolator that matches:
    - For cubic polynomial: position and velocity at endpoints
    - For quintic polynomial: position, velocity, and acceleration at endpoints
    
    The polynomial degree is automatically determined based on the provided constraints.
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
    Represents the time evolution of position and velocity for an object.
    
    Handles spatial trajectories including position and velocity interpolation,
    with optional support for periodic boundary conditions.
    
    Interpolation methods:
    1. Cubic Hermite spline for positions and velocities
    2. Quintic polynomial for positions, velocities and accelerations
    3. PCHIP for positions only (velocities estimated automatically)
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
        Evaluate trajectory at specified time(s).
        
        Parameters
        ----------
        t : float or array_like
            Time(s) at which to evaluate the trajectory
        wrap : bool, default=False
            If True, wrap positions to the periodic box [0, box_size).
            Only applies if box_size was specified.
        extrapolate : bool, default=False
            If True, extrapolate beyond the time range of data
            
        Returns
        -------
        tuple
            (positions, velocities) where:
            - positions: ndarray with shape (ndim,) or (N, ndim)
            - velocities: ndarray with shape (ndim,) or (N, ndim)
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
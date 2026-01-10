"""
galpos
======

A lightweight toolbox for representing galaxy motion (trajectory) and orientation
as continuous functions of time.

The public API centers around :class:`~galpos.GalaxyPoseTrajectory`, which bundles:

- :class:`~galpos.orbits.Trajectory` for position/velocity (and optional acceleration)
- :class:`~galpos.poses.Orientation` for rotation matrices over time

Typical workflow
----------------
1. Build a trajectory from discrete snapshots (t, pos, vel [, acc]).
2. (Optional) Build an orientation from rotation matrices or angular momentum.
3. Evaluate the galaxy state at arbitrary times.

Examples
--------
Create a trajectory with position and velocity and query an intermediate time:

>>> import numpy as np
>>> from galpos import GalaxyPoseTrajectory
>>> t = np.array([0.0, 1.0, 2.0])
>>> pos = np.array([[0, 0, 0],
...                 [1, 0, 0],
...                 [2, 0, 0]], dtype=float)
>>> vel = np.array([[1, 0, 0],
...                 [1, 0, 0],
...                 [1, 0, 0]], dtype=float)
>>> gpt = GalaxyPoseTrajectory(t, pos, vel)
>>> p, v, r = gpt(0.5)
>>> p.shape, v.shape, r
((3,), (3,), None)

Add an orientation using angular momentum directions:

>>> ang = np.array([[0, 0, 1],
...                 [0, 1, 1],
...                 [0, 1, 0]], dtype=float)
>>> gpt = GalaxyPoseTrajectory(t, pos, vel, angular_momentum=ang)
>>> p, v, rot = gpt(1.2)
>>> rot.shape
(3, 3)
"""

from typing import Optional, Union, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .orbits import Trajectory
from .poses import Orientation

__version__ = "0.1.1"

__author__ = "Shuai Lu (卢帅)"
__email__ = "lushuai@stu.xmu.edu.cn"


__all__ = ["GalaxyPoseTrajectory"]

class GalaxyPoseTrajectory:
    """
    Bundle of a galaxy's translational trajectory and (optional) orientation.

    This class is a thin convenience wrapper around:

    - :class:`galpos.orbits.Trajectory` (position/velocity/acceleration)
    - :class:`galpos.poses.Orientation` (rotation matrices vs. time)

    Parameters
    ----------
    times : array_like, shape (N,)
        Sample times (must be strictly increasing; unsorted inputs are sorted).
    positions : array_like, shape (N,) or (N, ndim)
        Sampled positions.
    velocities : array_like, optional, shape (N,) or (N, ndim)
        Sampled velocities. If omitted and `accelerations` is also omitted,
        the trajectory may fall back to a method that estimates derivatives.
    rotations : ndarray, optional, shape (N, 3, 3)
        Rotation matrices at the provided `orientation_times` (or `times` if
        `orientation_times` is not given).
    angular_momentum : ndarray, optional, shape (N, 3)
        Angular momentum vectors used to derive a "face-on" rotation matrix.
        Used when `rotations` is not provided.
    accelerations : array_like, optional, shape (N,) or (N, ndim)
        Sampled accelerations. If provided, a higher-order interpolator may be used.
    box_size : float, optional
        Periodic box size. If set, the internal interpolator unwraps positions to
        avoid discontinuities; outputs can be wrapped on demand.
    trajectory_method : {'spline', 'polynomial', 'pchip'}, default='spline'
        Interpolation strategy passed to :class:`galpos.orbits.Trajectory`.
    orientation_times : array_like, optional, shape (N,)
        Time grid for orientation samples. Defaults to `times`.

    Notes
    -----
    - Calling an instance returns ``(pos, vel, rot)``, where ``rot`` may be ``None``
      if no orientation was provided.
    - Use :meth:`final_state` to fetch the last sampled state without interpolation.

    Examples
    --------
    >>> import numpy as np
    >>> from galpos import GalaxyPoseTrajectory
    >>> t = np.array([0., 1., 2.])
    >>> pos = np.array([[0., 0., 0.],
    ...                 [1., 0., 0.],
    ...                 [2., 0., 0.]])
    >>> vel = np.array([[1., 0., 0.],
    ...                 [1., 0., 0.],
    ...                 [1., 0., 0.]])
    >>> g = GalaxyPoseTrajectory(t, pos, vel)
    >>> g(1.5)[0]
    array([1.5, 0. , 0. ])
    """

    def __init__(self,
                 times: ArrayLike,
                 positions: ArrayLike, 
                 velocities: Optional[ArrayLike] = None, 
                 rotations: Optional[np.ndarray] = None,
                 angular_momentum: Optional[np.ndarray] = None,
                 accelerations: Optional[ArrayLike] = None, 
                 box_size: Optional[float] = None, 
                 trajectory_method: str = 'spline',
                 orientation_times: Optional[ArrayLike] = None):
        """
        Parameters
        ----------
        times : array_like, shape (N,)
            Sample times (must be strictly increasing; unsorted inputs are sorted).
        positions : array_like, shape (N,) or (N, ndim)
            Sampled positions.
        velocities : array_like, optional, shape (N,) or (N, ndim)
            Sampled velocities. If omitted and `accelerations` is also omitted,
            the trajectory may fall back to a method that estimates derivatives.
        rotations : ndarray, optional, shape (N, 3, 3)
            Rotation matrices at the provided `orientation_times` (or `times` if
            `orientation_times` is not given).
        angular_momentum : ndarray, optional, shape (N, 3)
            Angular momentum vectors used to derive a "face-on" rotation matrix.
            Used when `rotations` is not provided.
        accelerations : array_like, optional, shape (N,) or (N, ndim)
            Sampled accelerations. If provided, a higher-order interpolator may be used.
        box_size : float, optional
            Periodic box size. If set, the internal interpolator unwraps positions to
            avoid discontinuities; outputs can be wrapped on demand.
        trajectory_method : {'spline', 'polynomial', 'pchip'}, default='spline'
            Interpolation strategy passed to :class:`galpos.orbits.Trajectory`.
        orientation_times : array_like, optional, shape (N,)
            Time grid for orientation samples. Defaults to `times`.
        """
        self.trajectory: Trajectory = Trajectory(
            times, positions, velocities, 
            accelerations, box_size, trajectory_method)

        self.orientation: Optional[Orientation] = None

        if rotations is not None or angular_momentum is not None:
            if orientation_times is None:
                orientation_times = times
            self.orientation = Orientation(
                orientation_times, rotations, angular_momentum
                )

    def __call__(self, 
                 t: Union[float, ArrayLike], 
                 wrap: bool = False, 
                 extrapolate: bool = False
                 ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Evaluate galaxy position/velocity/(optional) orientation at time(s) ``t``.

        Parameters
        ----------
        t : float or array_like
            Query time(s).
        wrap : bool, default=False
            If True and `box_size` was provided, wrap positions back into ``[0, L)``.
        extrapolate : bool, default=False
            If False, values outside the input time range typically become NaN
            (implementation-dependent for trajectory/orientation).

        Returns
        -------
        position : ndarray
            Shape ``(ndim,)`` for scalar ``t`` or ``(M, ndim)`` for array input.
        velocity : ndarray
            Same shape convention as `position`.
        rotation_matrix : ndarray or None
            Shape ``(3, 3)`` (scalar input) or ``(M, 3, 3)`` (array input),
            or ``None`` if no orientation model is available.

        Examples
        --------
        >>> import numpy as np
        >>> from galpos import GalaxyPoseTrajectory
        >>> t = np.array([0., 1., 2.])
        >>> pos = np.array([[0., 0., 0.],
        ...                 [1., 0., 0.],
        ...                 [2., 0., 0.]])
        >>> vel = np.ones_like(pos)
        >>> g = GalaxyPoseTrajectory(t, pos, vel)
        >>> p, v, r = g([0.5, 1.5])
        >>> p.shape, v.shape, r
        ((2, 3), (2, 3), None)
        """
        pos, vel = self.trajectory(t, wrap, extrapolate)
        
        if self.orientation is not None:
            rot = self.orientation(t, extrapolate)
            return pos, vel, rot
        else:
            return pos, vel, None
        
    def final_state(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Return the last *sampled* state (no interpolation).

        Returns
        -------
        position : ndarray
            Last sampled position.
        velocity : ndarray or None
            Last sampled velocity if present, else None.
        rotation_matrix : ndarray or None
            Last sampled rotation matrix if present, else None.
        """
        pos = self.trajectory.positions[-1]
        vel = self.trajectory.velocities[-1] if self.trajectory.velocities is not None else None
        rot = self.orientation.rotations[-1] if self.orientation is not None else None
        return pos, vel, rot

    def get_acceleration(self, 
                         t: Union[float, ArrayLike], 
                         extrapolate: bool = False
                         ) -> np.ndarray:
        """
        Evaluate acceleration at time(s) ``t``.

        Parameters
        ----------
        t : float or array_like
            Query time(s).
        extrapolate : bool, default=False
            If False, times outside the input range typically return NaN.

        Returns
        -------
        ndarray
            Acceleration with shape ``(ndim,)`` or ``(M, ndim)``.
        """
        return self.trajectory.get_acceleration(t, extrapolate)
    
    def __repr__(self) -> str:
        """Return a string representation of the GalaxyPoseTrajectory object."""
        try:
            n = len(self.trajectory.times)
            t = self.trajectory.times
            dim = self.trajectory.ndim
            if self.trajectory.box_size is not None:
                box = f", b={self.trajectory.box_size:.2g}"
            else:
                box = ""
            o = "+" if self.orientation is not None else "-"
            
            return f"GPT({n}p, {dim}D, t=[{t[0]:.3g}-{t[-1]:.3g}]{box}, o:{o})"
        except (IndexError, AttributeError):
            return "GPT(empty)"
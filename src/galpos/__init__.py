from typing import Optional, Union, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .orbits import Trajectory
from .poses import Orientation

__version__ = "0.1.0"


class GalaxyPoseTrajectory:
    """
    Represents the complete trajectory of a galaxy, including position, 
    velocity and orientation over time.
    
    This class combines the functionality of Trajectory and Orientation
    to provide a complete representation of a galaxy's motion and orientation.
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
        times : array_like
            Time array (N,)
        positions : array_like
            Position array (N,) or (N, ndim)
        velocities : array_like
            Velocity array (N,) or (N, ndim)
        rotations : array_like, optional
            Rotation array (N, 3, 3)
        angular_momentum : array_like, optional
            Angular momentum vectors (N, 3) representing disk orientation.
        box_size : float, optional
            Size of the periodic box. 
            If specified, positions will be wrapped to the box (0 ~ box_size)
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
        Evaluate galaxy state at time t.
        
        Parameters
        ----------
        t : float
            Time at which to evaluate
        wrap : bool, default=False
            If True, return positions wrapped to the periodic box
        extrapolate : bool, default=False
            If False, return NaN for times outside input range
        
        Returns
        -------
        tuple
            (position, velocity, rotation_matrix)
        """
        pos, vel = self.trajectory(t, wrap, extrapolate)
        
        if self.orientation is not None:
            rot = self.orientation(t, extrapolate)
            return pos, vel, rot
        else:
            return pos, vel, None

    def get_acceleration(self, 
                         t: Union[float, ArrayLike], 
                         extrapolate: bool = False
                         ) -> np.ndarray:
        """
        Get the acceleration of the galaxy at time t.

        Parameters
        ----------
        t : float
            Time at which to evaluate
        wrap : bool, default=False
            If True, return accelerations wrapped to the periodic box
        extrapolate : bool, default=False
            If False, return NaN for times outside input range

        Returns
        -------
        ndarray
            Acceleration vector (N,3) or (N,). NaN if outside input range
        """
        return self.trajectory.get_acceleration(t, extrapolate)
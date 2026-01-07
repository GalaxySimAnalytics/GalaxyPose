"""
Orientation and rotation utilities for 3D simulations.

This module provides classes and functions for representing and interpolating
3D orientations over time, using quaternions and rotation matrices.

Rotation representation (math)
------------------------------
A 3D rotation can be represented by:

- a rotation matrix R ∈ SO(3), or
- a unit quaternion q (‖q‖=1), with q and -q encoding the same rotation.

Quaternion interpolation is often numerically stable and avoids gimbal lock.

Interpolation options implemented here
--------------------------------------
1) Rotation-matrix input → quaternion SQUAD
   If you provide rotation matrices R_i at times t_i, we convert them to unit quaternions q_i
   and interpolate on each interval using SQUAD (spherical quadrangle), which is a smooth
   spline-like extension of SLERP.

   Let h = (t - t_i)/(t_{i+1}-t_i). Define:
   - u(h) = slerp(q_i, q_{i+1}, h)
   - v(h) = slerp(s_i, s_{i+1}, h)   (s_i are control points computed from neighboring q's)
   Then the SQUAD curve is:
   - q(h) = slerp(u(h), v(h), 2h(1-h))

   This typically gives smoother angular velocity than plain per-interval SLERP.

2) Angular-momentum input → direction SLERP + frame construction
   If you provide angular momentum vectors L_i, we normalize them into directions
   z_i = L_i / ‖L_i‖ (fallback to [0,0,1] if near-zero), interpolate directions by SLERP:
   - z(h) = slerp(z_i, z_{i+1}, h)

   Then we build a right-handed basis (x,y,z) using a reference up vector u (default [0,1,0]):
   - x = normalize(u × z)  (if nearly zero, choose an alternative up vector)
   - y = z × x
   and return R = [x; y; z] (rows as basis vectors).

When to use which mode
----------------------
- Use **rotations → SQUAD** when you already have reliable rotation matrices per snapshot
  (e.g., from a rigid-body solver or halo finder) and want smooth interpolation in time.
- Use **angular_momentum → direction SLERP** when you only care about aligning the galaxy
  disk/face-on direction and do not have a full rotation about that axis. This approach
  fixes the remaining degree of freedom using the chosen up-vector, so it is best when
  a consistent “up” reference makes sense for your analysis pipeline.
"""

from typing import Union, Optional, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


# =================== Quaternion Utility Functions ===================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions in [w,x,y,z] format.
    
    Parameters
    ----------
    q1 : np.ndarray
        First quaternion in format [w,x,y,z]
    q2 : np.ndarray
        Second quaternion in format [w,x,y,z]
        
    Returns
    -------
    np.ndarray
        Result of quaternion multiplication
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute inverse of quaternion [w,x,y,z].
    
    For unit quaternions, this is the same as conjugate.
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion in format [w,x,y,z]
    
    Returns
    -------
    np.ndarray
        Inverse quaternion
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_logarithm(q: np.ndarray) -> np.ndarray:
    """
    Compute logarithm of quaternion [w,x,y,z].
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion in format [w,x,y,z]
    
    Returns
    -------
    np.ndarray
        Logarithm of quaternion
    """
    v = q[1:]
    s = q[0]
    v_norm = np.linalg.norm(v)
    
    if v_norm < 1e-10:
        return np.zeros(4)
    
    theta = np.arctan2(v_norm, s)
    return np.array([0, *(theta * v / v_norm)])


def quaternion_exponential(q: np.ndarray) -> np.ndarray:
    """
    Compute exponential of quaternion [w,x,y,z].
    
    Parameters
    ----------
    q : np.ndarray
        Quaternion in format [w,x,y,z]
    
    Returns
    -------
    np.ndarray
        Exponential of quaternion
    """
    v = q[1:]
    v_norm = np.linalg.norm(v)
    
    if v_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    theta = v_norm
    return np.array([np.cos(theta), *(np.sin(theta) * v / v_norm)])


# =================== Rotation Utility Functions ===================
DEFAULT_UP_VECTOR = np.array([0.0, 1.0, 0.0])
def calculate_face_on_matrix(
    direction_vector: np.ndarray, 
    up_vector: np.ndarray = DEFAULT_UP_VECTOR
) -> np.ndarray:
    """
    Calculate rotation matrix to align a direction vector with the z-axis.
    
    Parameters
    ----------
    direction_vector : np.ndarray
        The vector to be aligned with the z-axis after transformation
    up_vector : np.ndarray, default=[0.0, 1.0, 0.0]
        Reference vector for determining the orientation around z-axis
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    # Normalize input vector
    direction = np.asarray(direction_vector)
    direction = direction / np.linalg.norm(direction)
    
    up = np.asarray(up_vector)
    
    # Check if direction is nearly parallel to up vector
    perpendicular1 = np.cross(up, direction)
    perp1_norm = np.linalg.norm(perpendicular1)
    
    # If nearly parallel, choose a different up vector
    if perp1_norm < 1e-6:
        # Find component with minimum absolute value to create new up vector
        min_idx = np.argmin(np.abs(direction))
        new_up = np.zeros(3)
        new_up[min_idx] = 1.0
        
        perpendicular1 = np.cross(new_up, direction)
        perp1_norm = np.linalg.norm(perpendicular1)
    
    # Normalize first perpendicular vector
    perpendicular1 = perpendicular1 / perp1_norm
    
    # Calculate second perpendicular vector to complete orthogonal basis
    perpendicular2 = np.cross(direction, perpendicular1)
    
    # Create rotation matrix by stacking basis vectors
    return np.stack((perpendicular1, perpendicular2, direction), axis=0)


# =================== Orientation Class ===================

class Orientation:
    """
    Time-dependent orientation model returning rotation matrices.

    Interpolation model (math)
    --------------------------
    Two initialization modes are supported:

    A) rotations (R_i) → SQUAD over unit quaternions q_i
       - Convert R_i → q_i (unit quaternions, with sign consistency handled during SLERP).
       - For t in [t_i, t_{i+1}], compute h in [0,1] and evaluate SQUAD:
         q(h) = slerp(slerp(q_i,q_{i+1},h), slerp(s_i,s_{i+1},h), 2h(1-h))
       - Convert q(h) back to R(t).

    B) angular momentum (L_i) → direction SLERP
       - Normalize L_i to z_i; interpolate z(h) on the unit sphere using SLERP.
       - Build R(t) by constructing a right-handed basis from (x,y,z) with a reference up.

    Conditions / practical notes
    ----------------------------
    - `times` must be strictly increasing after sorting.
    - Quaternion interpolation assumes rotations vary smoothly; discontinuous flips in the
      input R_i will still produce sharp changes.
    - Direction-based interpolation does not encode rotation about the disk axis; the chosen
      up-vector fixes that gauge and should be consistent across your dataset.

    Parameters
    ----------
    times : array_like, shape (N,)
        Sample times (must be strictly increasing; unsorted inputs are sorted).
    rotations : ndarray, optional, shape (N, 3, 3)
        Rotation matrices at `times`.
    angular_momentum : ndarray, optional, shape (N, 3)
        Angular momentum vectors at `times`. Used if `rotations` is None.

    Raises
    ------
    ValueError
        If neither `rotations` nor `angular_momentum` is provided.

    Examples
    --------
    From angular momentum directions:

    >>> import numpy as np
    >>> from galpos.poses import Orientation
    >>> t = np.array([0., 1., 2.])
    >>> L = np.array([[0., 0., 1.],
    ...               [0., 1., 1.],
    ...               [0., 1., 0.]])
    >>> o = Orientation(t, angular_momentum=L)
    >>> R = o(1.2)
    >>> R.shape
    (3, 3)

    From explicit rotation matrices:

    >>> import numpy as np
    >>> from galpos.poses import Orientation
    >>> t = np.array([0., 1., 2.])
    >>> Rm = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    >>> o = Orientation(t, rotations=Rm)
    >>> np.allclose(o(0.3), np.eye(3))
    True
    """
    
    def __init__(
        self, 
        times: ArrayLike,
        rotations: Optional[np.ndarray] = None,
        angular_momentum: Optional[np.ndarray] = None
    ):
        """
        Initialize an orientation trajectory.
        
        Parameters
        ----------
        times : array_like
            Time points array (N,)
        rotations : array_like, optional
            Rotation matrices array (N, 3, 3)
        angular_momentum : array_like, optional
            Angular momentum vectors (N, 3) representing disk orientation.
            Used if rotations is None to derive rotation matrices.
        
        Raises
        ------
        ValueError
            If times are not strictly increasing or if neither rotations
            nor angular_momentum is provided
        """
        times = np.asarray(times)
        
        # Sort by time if needed
        if not np.all(np.diff(times) > 0):
            idx = np.argsort(times)
            times = times[idx]
            if rotations is not None:
                rotations = np.asarray(rotations)[idx]
            if angular_momentum is not None:
                angular_momentum = np.asarray(angular_momentum)[idx]

        assert any(np.diff(times) > 0), "Times must be strictly increasing"
        
        self.times = times
        
        self.use_angmom_interp: bool = False
        self.rotations: np.ndarray
        self.quaternions: np.ndarray
        self.control_points: List[np.ndarray]
        self.angmom_directions: np.ndarray
        
        # Process rotation information based on provided inputs
        if rotations is not None:
            self._initialize_from_rotations(rotations)
        elif angular_momentum is not None:
            self._initialize_from_angular_momentum(angular_momentum)
        else:
            raise ValueError("Either rotations or angular_momentum must be provided")
        
    def _initialize_from_rotations(self, rotations: np.ndarray) -> None:
        """
        Initialize orientation from rotation matrices.
        
        Parameters
        ----------
        rotations : np.ndarray
            Rotation matrices array (N, 3, 3)
        """
        self.rotations = np.asarray(rotations)
        
        # Convert rotation matrices to quaternions
        self.quaternions = self._rotation_matrices_to_quaternions(self.rotations)
        self.quaternions = self._normalize_quaternions(self.quaternions)
        
        # Compute control points for SQUAD interpolation
        self.control_points = self._compute_squad_control_points(self.quaternions)

    def _initialize_from_angular_momentum(self, angular_momentum: np.ndarray) -> None:
        """
        Initialize orientation from angular momentum vectors.
        
        Parameters
        ----------
        angular_momentum : np.ndarray
            Angular momentum vectors (N, 3)
        """
        self.use_angmom_interp = True
        angmom = np.asarray(angular_momentum)
        
        # Normalize angular momentum vectors
        norms = np.linalg.norm(angmom, axis=1)
        valid_mask = norms > 1e-10
        
        # Create array of normalized direction vectors
        self.angmom_directions = np.zeros_like(angmom)
        self.angmom_directions[valid_mask] = (
            angmom[valid_mask] / norms[valid_mask, np.newaxis]
            )

        # Set default direction [0,0,1] for invalid vectors
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            self.angmom_directions[invalid_mask] = [0, 0, 1]
        
        # Calculate rotation matrices from angular momentum directions
        self.rotations = np.array([
            calculate_face_on_matrix(direction) 
            for direction in self.angmom_directions
        ])
        
    def __call__(
        self, 
        t: Union[float, np.ndarray], 
        extrapolate: bool = False
    ) -> np.ndarray:
        """
        Evaluate rotation matrix/matrices at time(s) ``t``.

        Parameters
        ----------
        t : float or ndarray
            Query time(s).
        extrapolate : bool, default=False
            If False, out-of-range times return NaN matrices.

        Returns
        -------
        ndarray
            For scalar input: shape ``(3, 3)``.
            For array input: shape ``(M, 3, 3)``.
        """
        # Convert input to array for consistent processing
        t_array = np.atleast_1d(t)
        scalar_input = np.isscalar(t)
        
        results = np.zeros((len(t_array), 3, 3))
        
        if not extrapolate:
            out_of_bounds = (t_array < self.times[0]) | (t_array > self.times[-1])
            if np.any(out_of_bounds):
                results[out_of_bounds] = np.full((3, 3), np.nan)


        valid_indices = (np.logical_not(out_of_bounds) 
                         if not extrapolate 
                         else np.ones_like(t_array, dtype=bool))
        
        valid_times = t_array[valid_indices]
        
        if len(valid_times) == 0:
            return results[0] if scalar_input else results
        
        indices = np.searchsorted(self.times, valid_times) - 1
        indices = np.clip(indices, 0, len(self.times) - 2)
        
        t0 = self.times[indices]
        t1 = self.times[indices + 1]
        h = (valid_times - t0) / (t1 - t0)
        
        if self.use_angmom_interp:
            results[valid_indices] = self._interpolate_directions(indices, h)
        else:
            results[valid_indices] = self._squad_interpolation(indices, h)
        
        return results[0] if scalar_input else results

    def _interpolate_directions(self, indices: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Batch direction vector interpolation (SLERP).
        
        Parameters
        ----------
        indices : np.ndarray
            Index array indicating time intervals for each point
        h : np.ndarray
            Normalized position [0,1] within each interval
            
        Returns
        -------
        np.ndarray
            Rotation matrices array with shape (len(indices), 3, 3)
        """
        # Get interval start and end direction vectors
        v0 = self.angmom_directions[indices]
        v1 = self.angmom_directions[indices + 1]
        
        dot = np.sum(v0 * v1, axis=1)
        
        # Ensure shortest path interpolation
        neg_mask = dot < 0
        if np.any(neg_mask):
            v1[neg_mask] = -v1[neg_mask]
            dot[neg_mask] = -dot[neg_mask]
        
        results = np.zeros((len(indices), 3))
        
        # Use linear interpolation for nearly parallel vectors
        linear_mask = dot > 0.9999
        if np.any(linear_mask):
            res_lin = (v0[linear_mask] + 
                       h[linear_mask, np.newaxis] * (v1[linear_mask] - v0[linear_mask]))
            norm = np.linalg.norm(res_lin, axis=1, keepdims=True)
            results[linear_mask] = res_lin / norm
        
        # Use full SLERP for other vectors
        slerp_mask = ~linear_mask
        if np.any(slerp_mask):
            h_slerp = h[slerp_mask]
            dot_slerp = dot[slerp_mask]
            
            theta = np.arccos(np.clip(dot_slerp, -1.0, 1.0))
            sin_theta = np.sin(theta)
            
            a = np.sin((1.0 - h_slerp) * theta) / sin_theta
            b = np.sin(h_slerp * theta) / sin_theta
            
            res_slerp = (a[:, np.newaxis] * v0[slerp_mask] + 
                        b[:, np.newaxis] * v1[slerp_mask])
            
            # Normalize results
            norm = np.linalg.norm(res_slerp, axis=1, keepdims=True)
            results[slerp_mask] = res_slerp / norm
        
        # Batch convert direction vectors to rotation matrices
        rotations = np.zeros((len(indices), 3, 3))
        
        # Normalized direction vectors (z-axis)
        z_axis = results
        
        # Generate orthogonal basis vectors
        up_vector = np.array([0.0, 1.0, 0.0])
        
        # Calculate first orthogonal vector (x-axis)
        x_axis = np.cross(np.tile(up_vector, (len(indices), 1)), z_axis)
        x_norms = np.linalg.norm(x_axis, axis=1)
        
        # Handle special case of vectors parallel to up vector
        parallel_mask = x_norms < 1e-6
        if np.any(parallel_mask):
            # Choose a different up vector for these cases
            alt_up = np.array([1.0, 0.0, 0.0])  
            x_axis[parallel_mask] = np.cross(
                np.tile(alt_up, (np.sum(parallel_mask), 1)), 
                z_axis[parallel_mask])
            x_norms[parallel_mask] = np.linalg.norm(x_axis[parallel_mask], axis=1)
        
        # Normalize x-axis
        x_axis = x_axis / x_norms[:, np.newaxis]
        
        # Calculate y-axis (ensure right-handed coordinate system)
        y_axis = np.cross(z_axis, x_axis)
        
        # Fill rotation matrices
        rotations[:, 0, :] = x_axis
        rotations[:, 1, :] = y_axis
        rotations[:, 2, :] = z_axis
        
        return rotations

    def _squad_interpolation(self, indices: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Batch Spherical Quadrangle (SQUAD) interpolation for quaternions.
        
        Parameters
        ----------
        indices : np.ndarray
            Index array indicating time intervals for each point
        h : np.ndarray
            Normalized position [0,1] within each interval
            
        Returns
        -------
        np.ndarray
            Rotation matrices array with shape (len(indices), 3, 3)
        """
        # Get quaternions and control points for intervals
        q0 = self.quaternions[indices]
        q1 = self.quaternions[indices + 1]
        s0 = np.array([self.control_points[i] for i in indices])
        s1 = np.array([self.control_points[i+1] for i in indices])
        
        # Batch SLERP calculation
        def batch_slerp(qa, qb, t):
            """Batch spherical linear interpolation"""
            # Ensure unit quaternions
            qa_norm = np.linalg.norm(qa, axis=1, keepdims=True)
            qb_norm = np.linalg.norm(qb, axis=1, keepdims=True)
            
            qa = qa / qa_norm
            qb = qb / qb_norm
            
            # Calculate angle between quaternions
            dot = np.sum(qa * qb, axis=1)
            
            # Ensure shortest path
            neg_mask = dot < 0
            if np.any(neg_mask):
                qb[neg_mask] = -qb[neg_mask]
                dot[neg_mask] = -dot[neg_mask]
            
            results = np.zeros_like(qa)
            
            # Use linear interpolation for nearly parallel quaternions
            DOT_THRESHOLD = 0.9995
            linear_mask = dot > DOT_THRESHOLD
            
            if np.any(linear_mask):
                t_lin = t[linear_mask, np.newaxis]
                res_lin = qa[linear_mask] + t_lin * (qb[linear_mask] - qa[linear_mask])
                norm_lin = np.linalg.norm(res_lin, axis=1, keepdims=True)
                results[linear_mask] = res_lin / norm_lin
            
            # Use full SLERP for other quaternions
            slerp_mask = ~linear_mask
            if np.any(slerp_mask):
                t_slerp = t[slerp_mask]
                dot_slerp = dot[slerp_mask]
                
                theta_0 = np.arccos(np.clip(dot_slerp, -1.0, 1.0))
                theta = theta_0 * t_slerp
                
                sin_theta = np.sin(theta)
                sin_theta_0 = np.sin(theta_0)
                
                s0 = np.cos(theta) - dot_slerp * sin_theta / sin_theta_0
                s1 = sin_theta / sin_theta_0
                
                results[slerp_mask] = (s0[:, np.newaxis] * qa[slerp_mask] + 
                                    s1[:, np.newaxis] * qb[slerp_mask])
            
            return results
        
        # SQUAD interpolation
        slerp1 = batch_slerp(q0, q1, h)
        slerp2 = batch_slerp(s0, s1, h)
        
        # Final SQUAD result
        squad_result = batch_slerp(slerp1, slerp2, 2 * h * (1 - h))
        
        # Batch convert quaternions to rotation matrices
        # Convert from [w,x,y,z] format to [x,y,z,w] format for scipy
        scipy_quats = squad_result[:, [1, 2, 3, 0]]
        rotation_matrices = Rotation.from_quat(scipy_quats).as_matrix()
        
        return rotation_matrices
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical Linear Interpolation between quaternions.
        
        Parameters
        ----------
        q1 : np.ndarray
            First quaternion in format [w,x,y,z]
        q2 : np.ndarray
            Second quaternion in format [w,x,y,z]
        t : float
            Interpolation parameter in range [0,1]
            
        Returns
        -------
        np.ndarray
            Interpolated quaternion
        """
        # Ensure unit quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate angle between quaternions
        dot: float = np.sum(q1 * q2)
        
        # Choose shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot
            
        # Linear interpolation for very close quaternions
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Full SLERP
        theta_0 = np.arccos(dot)  
        theta = theta_0 * t
        
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def _compute_squad_control_points(
        self, 
        quaternions: np.ndarray, 
    ) -> List[np.ndarray]:
        """
        Compute control points for SQUAD quaternion interpolation.
        
        Parameters
        ----------
        quaternions : np.ndarray
            Array of quaternions
            
        Returns
        -------
        List[np.ndarray]
            Control points for SQUAD interpolation
        """
        num_points = len(quaternions)
        control_points = [np.zeros(4) for _ in range(num_points)]
        
        # Calculate interior control points
        for i in range(1, num_points - 1):
            a = quaternion_logarithm(
                quaternion_multiply(
                    quaternion_inverse(quaternions[i]), quaternions[i+1]
                )
            )
            b = quaternion_logarithm(
                quaternion_multiply(
                    quaternion_inverse(quaternions[i]), quaternions[i-1]
                )
            )
            control_points[i] = quaternion_multiply(
                quaternions[i], 
                quaternion_exponential(-0.25 * (a + b))
            )
        
        # Handle endpoints
        if num_points > 2:
            # Start point
            a_0 = quaternion_logarithm(
                quaternion_multiply(
                    quaternion_inverse(quaternions[0]), quaternions[1]
                    )
            )
            control_points[0] = quaternion_multiply(
                quaternions[0],
                quaternion_exponential(-0.5 * a_0)
            )
            
            # End point
            b_n = quaternion_logarithm(
                quaternion_multiply(
                    quaternion_inverse(quaternions[-1]), quaternions[-2]
                    )
            )
            control_points[-1] = quaternion_multiply(
                quaternions[-1],
                quaternion_exponential(-0.5 * b_n)
            )
        else:
            # For only two points, use the quaternions themselves as control points
            control_points[0] = quaternions[0]
            control_points[-1] = quaternions[-1]
        
        return control_points
    
    def _normalize_quaternions(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Normalize quaternions to unit length.
        
        Parameters
        ----------
        quaternions : np.ndarray
            Array of quaternions to normalize
            
        Returns
        -------
        np.ndarray
            Array of normalized quaternions
        """
        norms = np.sqrt(np.sum(quaternions * quaternions, axis=1))
        mask = norms > 1e-10
        
        normalized = quaternions.copy()
        normalized[mask] = quaternions[mask] / norms[mask, np.newaxis]
        
        return normalized

    @staticmethod
    def _rotation_matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to quaternions [w,x,y,z].
        
        Parameters
        ----------
        matrices : np.ndarray
            Array of 3x3 rotation matrices
            
        Returns
        -------
        np.ndarray
            Array of quaternions in [w,x,y,z] format
        """
        # Convert using scipy and reorder components from [x,y,z,w] to [w,x,y,z]
        return Rotation.from_matrix(matrices).as_quat()[:, [3, 0, 1, 2]]
    
    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [w,x,y,z] to rotation matrix.
        
        Parameters
        ----------
        quaternion : np.ndarray
            Quaternion in [w,x,y,z] format
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        # Reorder components from [w,x,y,z] to [x,y,z,w] for scipy
        q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        return Rotation.from_quat(q).as_matrix()
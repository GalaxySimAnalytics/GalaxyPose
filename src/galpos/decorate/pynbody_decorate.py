"""Pynbody integration (optional).

This module provides `StarBirth` and `make_star_birth`, which align stellar birth
positions/velocities into a host-galaxy frame described by a
`galpos.GalaxyPoseTrajectory`.

It requires `pynbody` and is imported lazily via `galpos.decorate`.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
from pynbody.snapshot import SimSnap
from pynbody.array import SimArray
from pynbody.units import Unit

from galpos import GalaxyPoseTrajectory



__all__ = ["make_star_birth", "StarBirth"]


def units_transform(sim: SimSnap, array_name: str, new_units: Union[str, Unit]) -> None:
    """
    In-place unit conversion for a named array inside a `pynbody` SimSnap.

    Parameters
    ----------
    sim : pynbody.snapshot.SimSnap
        The snapshot containing the target array.
    array_name : str
        Array key, e.g. ``'pos'`` or ``'vel'``.
    new_units : str or pynbody.units.Unit
        Target unit string or Unit.

    Returns
    -------
    None
        The array is modified in-place
    """
    array = sim[array_name]
    ratio = array.units.ratio(new_units, **(sim.conversion_context()))
    # `pynbody` accepts unit strings here; avoids relying on Unit(...) constructor typing.
    array.units = new_units
    if np.isscalar(ratio):
        # Simple scalar multiplication for all shapes
        array[:] = array.view(np.ndarray) * ratio
    else:
        # Use broadcasting for 1D, 2D and higher dimensional arrays
        broadcast_shape = (len(ratio),) + (1,) * (array.ndim - 1)
        reshaped_ratio = ratio.reshape(broadcast_shape)
        array[:] = array.view(np.ndarray) * reshaped_ratio



class StarBirth(SimSnap):
    """
    A `pynbody` snapshot representing stellar birth properties aligned to a host galaxy.

    The object stores birth ``pos``, ``vel``, ``mass`` and ``tform`` arrays and a
    reference to the host :class:`galpos.GalaxyPoseTrajectory`.

    Parameters
    ----------
    pos, vel, mass, time : pynbody.array.SimArray
        Birth properties.
    scale_factor : ndarray
        Scale factor at formation times. Stored in ``self.properties['a']``.
    galaxy_orbit : galpos.GalaxyPoseTrajectory
        Host trajectory/orientation used for alignment.

    See Also
    --------
    make_star_birth : Convenience constructor producing a :class:`StarBirth`.
    """
    def __init__(self, pos: SimArray, vel: SimArray, mass: SimArray, 
                 time: SimArray, scale_factor: np.ndarray, 
                 galaxy_orbit: GalaxyPoseTrajectory) -> None:
        
        # Initialize base SimSnap
        SimSnap.__init__(self)
                    
        # Set the number of particles
        self._num_particles = len(pos)
        
        # Set up the family slice for star particles
        from pynbody import family
        self._family_slice[family.star] = slice(0, len(pos))
        
        # Create the arrays
        self._create_array('pos', ndim=3, dtype=pos.dtype)
        self._create_array('vel', ndim=3, dtype=vel.dtype)
        self._create_array('mass', ndim=1, dtype=mass.dtype)
        self._create_array('tform', ndim=1, dtype=time.dtype)
        
        # Set the array values
        self['pos'][:] = pos
        self['pos'].units = pos.units
        self['vel'][:] = vel
        self['vel'].units = vel.units
        self['mass'][:] = mass
        self['mass'].units = mass.units
        self['tform'][:] = time
        self['tform'].units = time.units

        # Store the scale factor
        self.properties['a'] = scale_factor.view(np.ndarray)

        # Store the galaxy orbit information
        self.galaxy_orbit = galaxy_orbit

        # Track alignment state
        self.__already_centered = False
        self.__already_oriented = False
        
        self._filename = self._get_filename_with_status()
        self._decorate()

    def _get_filename_with_status(self) -> str:
        """Generate filename with alignment status information."""
        status = []
        if self.__already_centered:
            status.append("centered")
        if self.__already_oriented:
            status.append("oriented")
        
        status_str = f" [{','.join(status)}]" if status else ""
        return f"{repr(self.galaxy_orbit)}{status_str}"

    def align_with_galaxy(self, orientation_align: bool = True) -> "StarBirth":
        """
        Center (and optionally rotate) star birth coordinates into the host frame.

        Steps
        -----
        1. Subtract host position/velocity at each particle's formation time.
        2. If host orientation is available and `orientation_align=True`,
           rotate positions/velocities by the host rotation matrix at formation time.

        Parameters
        ----------
        orientation_align : bool, default=True
            Whether to apply orientation alignment in addition to centering.

        Returns
        -------
        StarBirth
            Returns ``self`` to allow chaining.

        Examples
        --------
        >>> sb = sb.align_with_galaxy(orientation_align=True)
        """

        if self.__already_centered and (self.__already_oriented or not orientation_align):
            warnings.warn(
                "StarBirth is already aligned; returning self",
                RuntimeWarning,
                stacklevel=2,
            )
            return self

        if not self.__already_centered:

            pos, vel = self.galaxy_orbit.trajectory(
                self.s['tform'].in_units('Gyr'), wrap=True)

            units_transform(self.s, "pos", "a kpc")
            self.s['pos'] = self.s['pos'] - pos

            units_transform(self.s, "vel", "a kpc Gyr**-1")
            self.s['vel'] = self.s['vel'] - vel

            units_transform(self.s, "pos", "kpc")
            units_transform(self.s, "vel", "km s**-1")

            self.__already_centered = True
            self._filename = self._get_filename_with_status()

        if (orientation_align and 
            not self.__already_oriented):
            if self.galaxy_orbit.orientation is not None:
                trans = self.galaxy_orbit.orientation(self.s['tform'].in_units('Gyr'))
                
                self.s['pos'][:] = np.einsum("ij,ikj->ik", 
                                             self.s['pos'].view(np.ndarray), trans)
                
                self.s['vel'][:] = np.einsum("ij,ikj->ik", 
                                             self.s['vel'].view(np.ndarray), trans)
                self.__already_oriented = True
                self._filename = self._get_filename_with_status()
            else:
                warnings.warn(
                    "Galaxy orientation not available; only centering was applied",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self
    
    def final_state(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return self.galaxy_orbit.final_state()
    
    def _register_transformation(self, t: Any) -> None:
        warnings.warn(
            ("StarBirth usually does not require any coordinate transformations, "
            "the only available transformation method is align_with_galaxy"),
            UserWarning, stacklevel=2
            )
        super()._register_transformation(t)

def make_star_birth(galaxy_orbit: GalaxyPoseTrajectory, 
               birth_time: np.ndarray, 
               birth_pos: np.ndarray, 
               birth_velocity: np.ndarray, 
               mass: np.ndarray, 
               scale_factor: np.ndarray,
               birth_pos_units: str = "kpc",
               birth_time_units: str = "Gyr",
               birth_velocity_units: str = "kpc Gyr**-1",
               mass_units: str = "Msol",
               cosmology_params: Optional[dict] = None) -> StarBirth:
    """
    Build a :class:`StarBirth` snapshot from raw birth arrays.

    Parameters
    ----------
    galaxy_orbit : galpos.GalaxyPoseTrajectory
        Host galaxy trajectory/orientation.
    birth_time : ndarray, shape (N,)
        Formation times.
    birth_pos : ndarray, shape (N, 3)
        Formation positions.
    birth_velocity : ndarray, shape (N, 3)
        Formation velocities.
    mass : ndarray, shape (N,)
        Stellar masses (birth or current, depending on your upstream choice).
    scale_factor : ndarray, shape (N,)
        Scale factor at formation time.
    birth_pos_units, birth_time_units, birth_velocity_units, mass_units : str
        Unit strings used to construct `pynbody` SimArray objects.
    cosmology_params : dict, optional
        If provided, copied into ``StarBirth.properties``.

    Returns
    -------
    StarBirth
        Snapshot holding the birth properties.

    Notes
    -----
    Particles with ``scale_factor`` outside ``(0, 1]`` are removed.

    Examples
    --------
    See module-level example.
    """
    sel = (scale_factor > 0) & (scale_factor <= 1)
    if not sel.all():
        birth_pos = birth_pos[sel]
        birth_time = birth_time[sel]
        birth_velocity = birth_velocity[sel]
        mass = mass[sel]
        scale_factor = scale_factor[sel]
        np_remove = int(np.size(sel) - len(birth_pos))
        warnings.warn(
            f"Removed {np_remove} particles due to invalid scale factors.",
            RuntimeWarning,
            stacklevel=2,
        )
        
    star = StarBirth(
        pos=SimArray(birth_pos, birth_pos_units),
        vel=SimArray(birth_velocity, birth_velocity_units),
        mass=SimArray(mass, mass_units),
        time=SimArray(birth_time, birth_time_units),
        scale_factor=scale_factor,
        galaxy_orbit=galaxy_orbit
    )
    if cosmology_params is not None:
        star.properties.update(cosmology_params)

    return star

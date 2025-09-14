""" 
This module aligns stellar birth positions with galaxy trajectories and orientations.
It provides functionality to decorate particle information using Pynbody.

The module contains utilities for creating and manipulating star birth data
in relation to galaxy pose trajectories, allowing for proper alignment
of stellar positions and velocities with their host galaxies.
"""
import warnings
from typing import Union, Any, Optional
from typing_extensions import Self

import numpy as np

from pynbody.snapshot import SimSnap
from pynbody.array import SimArray
from pynbody.units import Unit

from . import GalaxyPoseTrajectory



__all__ = ["make_star_birth", "StarBirth"]


def units_transform(sim: SimSnap, array_name: str, new_units: Union[str, Unit]) -> None:
    """
    Transform an array in a simulation to new units.
    
    Parameters
    ----------
    sim : Any
        Simulation object containing the array to transform
    array_name : str
        Name of the array to transform
    new_units : str
        Target units to transform to
        
    Returns
    -------
    None
        The array is modified in-place
    """
    array = sim[array_name]
    ratio = array.units.ratio(new_units, **(sim.conversion_context()))
    array.units = Unit(new_units)
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
    Class representing stellar birth data aligned with galaxy trajectory.

    Inherits from pynbody.snapshot.SimSnap and provides methods to align
    stellar positions and velocities with their host galaxy's trajectory
    and orientation.
    
    Parameters
    ----------
    pos : SimArray
        Birth positions of stars
    vel : SimArray
        Birth velocities of stars
    mass : SimArray
        Masses of stars
    time : SimArray
        Formation times of stars
    scale_factor : np.ndarray
        Scale factors at formation times
    galaxy_orbit : GalaxyPoseTrajectory
        Galaxy trajectory and orientation data
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

    def align_with_galaxy(self, orientation_align: bool = True) -> Self:
        """
        Align star positions and velocities with their host galaxy.
        
        This method performs two operations:
        1. Center positions and velocities relative to the galaxy's position 
        and velocity
        2. (Optional) Orient positions and velocities according to the 
        galaxy's orientation
        
        Parameters
        ----------
        orientation_align : bool, default=True
            Whether to align with galaxy orientation in addition to position
            
        Returns
        -------
        None
            The positions and velocities are modified in-place
        """

        if (self.__already_centered and 
            (self.__already_oriented or not orientation_align)):
            print("Already centered and oriented" 
                  if self.__already_oriented else "Already centered")
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
                print("Galaxy orientation not available")
        return self
    
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
    Create a StarBirth object from stellar birth data.
    
    Parameters
    ----------
    galaxy_orbit : GalaxyPoseTrajectory
        Galaxy trajectory and orientation data
    birth_pos : np.ndarray
        Birth positions of stars
    birth_time : np.ndarray
        Formation times of stars
    birth_velocity : np.ndarray
        Birth velocities of stars
    mass : np.ndarray
        Masses of stars
    scale_factor : np.ndarray
        Scale factors at formation times
    birth_pos_units : str, default="kpc"
        Units for birth positions
    birth_time_units : str, default="Gyr"
        Units for birth times
    birth_velocity_units : str, default="kpc Gyr**-1"
        Units for birth velocities
    mass_units : str, default="Msol"
        Units for masses
    cosmology_params : dict, optional
        Cosmology parameters to include in the StarBirth object

    Returns
    -------
    StarBirth
        A new StarBirth object with the provided data
        
    Notes
    -----
    Invalid particles with scale factors outside (0,1] will be removed.
    """
    sel = (scale_factor > 0) & (scale_factor <= 1)
    if not sel.all():
        birth_pos = birth_pos[sel]
        birth_time = birth_time[sel]
        birth_velocity = birth_velocity[sel]
        mass = mass[sel]
        scale_factor = scale_factor[sel]
        np_remove = len(sel) - len(birth_pos)
        print(f"Removed {np_remove} particles due to invalid scale factors.")
        
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

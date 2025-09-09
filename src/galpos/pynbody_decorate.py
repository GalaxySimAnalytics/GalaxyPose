""" 
This module aligns stellar birth positions with galaxy trajectories and orientations.
It provides functionality to decorate particle information using Pynbody.

The module contains utilities for creating and manipulating star birth data
in relation to galaxy pose trajectories, allowing for proper alignment
of stellar positions and velocities with their host galaxies.
"""

import numpy as np
from typing import Union

from . import GalaxyPoseTrajectory

from pynbody.snapshot import new, SubSnap
from pynbody.array import SimArray
from pynbody.units import Unit



__all__ = ["make_star_birth", "StarBirth"]


def units_transform(sim: SubSnap, array_name: str, new_units: Union[str, Unit]) -> None:
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
        array[:] = array.view(np.ndarray) * ratio
    else:
        array[:] = np.einsum("ij,i->ij", array.view(np.ndarray), ratio)



class StarBirth(SubSnap):
    """
    Class representing stellar birth data aligned with galaxy trajectory.
    
    Inherits from pynbody.snapshot.SubSnap and provides methods to align
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

        star = new(star = len(pos))
        star.s['pos'] = pos
        star.s['vel'] = vel
        star.s['mass'] = mass
        star.s['tform'] = time
        star.properties['a'] = scale_factor

        SubSnap.__init__(self, star, slice(len(star)))
        self.galaxy_orbit = galaxy_orbit
        self.__already_centered = False
        self.__already_oriented = False

    def align_with_galaxy(self, orientation_align: bool = True) -> None:
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
            return

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


        if (orientation_align and 
            not self.__already_oriented):
            if self.galaxy_orbit.orientation is not None:
                trans = self.galaxy_orbit.orientation(self.s['tform'].in_units('Gyr'))
                
                self.s['pos'][:] = np.einsum("ij,ikj->ik", 
                                             self.s['pos'].view(np.ndarray), trans)
                
                self.s['vel'][:] = np.einsum("ij,ikj->ik", 
                                             self.s['vel'].view(np.ndarray), trans)
                self.__already_oriented = True
            else:
                print("Galaxy orientation not available")



def make_star_birth(galaxy_orbit: GalaxyPoseTrajectory, 
               birth_time: np.ndarray, 
               birth_pos: np.ndarray, 
               birth_velocity: np.ndarray, 
               mass: np.ndarray, 
               scale_factor: np.ndarray,
               birth_pos_units: str = "kpc",
               birth_time_units: str = "Gyr",
               birth_velocity_units: str = "kpc Gyr**-1",
               mass_units: str = "Msol") -> StarBirth:
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

    return star

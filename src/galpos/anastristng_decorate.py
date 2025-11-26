from typing import Any, Optional, Union

import numpy as np
from AnastrisTNG.TNGsimulation import Snapshot

from pynbody.array import SimArray

from .import GalaxyPoseTrajectory
from .pynbody_decorate import make_star_birth as _make_star_birth
from .pynbody_decorate import StarBirth, Unit



def units_transform(array: SimArray, new_units: Union[str, Unit], **convertion_context: Any) -> SimArray:
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
    ratio = array.units.ratio(new_units, **convertion_context)
    array.units = Unit(new_units)
    if np.isscalar(ratio):
        # Simple scalar multiplication for all shapes
        new_arr = array.view(np.ndarray) * ratio
    else:
        # Use broadcasting for 1D, 2D and higher dimensional arrays
        broadcast_shape = (len(ratio),) + (1,) * (array.ndim - 1)
        reshaped_ratio = ratio.reshape(broadcast_shape)
        new_arr = array.view(np.ndarray) * reshaped_ratio
    new_arr = SimArray(new_arr, new_units)
    return new_arr

def make_star_birth(snapshot: Snapshot, 
                    ID: int, 
                    issubhalo: bool = True, 
                    host_t : Optional[SimArray] = None,
                    host_pos: Optional[SimArray] = None,
                    host_vel: Optional[SimArray] = None,
                    angular_momentum: Optional[Union[np.ndarray, SimArray]] = None,
                    useCM: bool = False,
                    useBirthmass: bool = False,
                    ) -> StarBirth:
    """
    Create a StarBirth object from a snapshot and particle ID.
    This function extracts the necessary information from the snapshot
    and constructs a StarBirth object representing the stellar birth
    properties of the specified particle.

    Parameters
    ----------
    snapshot : Snapshot
        The snapshot from which to extract particle data
    ID : int
        The ID to extract
    issubhalo : bool, default=True
        Whether the ID corresponds to a subhalo or a main halo
    host_t : SimArray, optional
        User-provided host galaxy's time array
    host_pos : SimArray, optional
        User-provided host galaxy's position array
    host_vel : SimArray, optional
        User-provided host galaxy's velocity array
    angular_momentum : Union[np.ndarray, SimArray], optional
        User-provided angular momentum of the host galaxy
    useCM : bool, default=False
        Whether to use the center of mass for positioning
    useBirthmass : bool, default=False
        Whether to use the birth mass instead of the current mass

    Returns
    -------
    StarBirth
        A StarBirth object containing the extracted properties
    """
    originfield = snapshot.load_particle_para['star_fields'].copy()
    snapshot.load_particle_para['star_fields'] = [
        'Coordinates', 'Velocities', 'Masses', 'ParticleIDs',
        'GFM_StellarFormationTime', 'GFM_InitialMass', 'BirthPos', 'BirthVel']
    
    # Check if user is providing custom pose and orientation
    user_pose = host_t is not None and host_pos is not None and host_vel is not None
    user_orient = host_t is not None and angular_momentum is not None
    
    # Load particles based on group type
    if issubhalo:
        PT = snapshot.load_particle(
            ID, groupType = 'Subhalo', decorate = False, order = 'star',
            )
        
        # Only load evolution data if needed
        if not (user_pose and user_orient):
            evo = snapshot.galaxy_evolution(
                ID, 
                ['SubhaloPos', 'SubhaloVel', 'SubhaloSpin','SubhaloCM'], 
                physical_units=False
            )
    else:
        PT = snapshot.load_particle(
            ID, groupType = 'Halo', decorate = False, order = 'star',
        )
        
        # Only load evolution data if needed
        if not (user_pose and user_orient):
            evo = snapshot.halo_evolution(
                ID, physical_units=False
            )
    # Determine position and velocity information
    if user_pose:
        times = host_t
        pos = host_pos
        vel = host_vel
    else:
        times = evo['t']
        if issubhalo:
            pos = evo['SubhaloCM'] if useCM else evo['SubhaloPos']
        else:
            pos = evo['GroupCM'] if useCM else evo['GroupPos']
        vel = evo['SubhaloVel'] if issubhalo else evo['GroupVel']
        times = times.in_units("Gyr")
        pos = units_transform(pos, "a kpc", a = evo['a'], h=snapshot.properties['h'])
        vel = units_transform(vel, "a kpc Gyr**-1", a = evo['a'], h=snapshot.properties['h'])
    
    
    # Determine orientation information
    if user_orient:
        orientation_times = host_t
        ang_mom = angular_momentum
    else:
        if issubhalo:
            orientation_times = evo['t']
            ang_mom = evo['SubhaloSpin']
        else:
            orientation_times = None
            ang_mom = None


    # Create orbit trajectory
    orbit = GalaxyPoseTrajectory(
        times, pos, vel, 
        box_size = float(snapshot.properties['boxsize'].in_units("a kpc", **snapshot.conversion_context())),
        angular_momentum=ang_mom, orientation_times=orientation_times)
    
    # Extract particle properties
    birth_time = PT['tform']
    birth_pos = PT['BirthPos']
    birth_velocity = PT['BirthVel']
    mass = PT['GFM_InitialMass'] if useBirthmass else PT['mass']
    scale_factor = PT['aform']
    scale_factor = scale_factor.view(np.ndarray)
    
    # Restore original star fields
    snapshot.load_particle_para['star_fields'] = originfield
    
    # Create and return StarBirth object
    return _make_star_birth(
        orbit, birth_time, birth_pos, birth_velocity, mass, scale_factor,
        birth_pos_units=birth_pos.units, birth_time_units=birth_time.units, 
        birth_velocity_units=birth_velocity.units, mass_units=mass.units, 
        cosmology_params=PT.properties["cosmology"])

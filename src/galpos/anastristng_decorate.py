"""
anastristng_decorate
====================

Integration helpers for building :class:`galpos.pynbody_decorate.StarBirth`
objects from `AnastrisTNG` snapshots.

This module extracts:

- host galaxy evolution (pos/vel/spin) from AnastrisTNG
- particle birth properties (BirthPos/BirthVel/tform/aform)
and then delegates to :func:`galpos.pynbody_decorate.make_star_birth`.

Examples
--------
Typical usage pattern (requires AnastrisTNG + pynbody):

>>> from AnastrisTNG.TNGsimulation import Snapshot
>>> from galpos.anastristng_decorate import make_star_birth
>>> snap = Snapshot(...)
>>> sb = make_star_birth(snap, ID=12345, issubhalo=True)
>>> sb = sb.align_with_galaxy()
"""
from typing import Any, Optional, Union

import numpy as np
from AnastrisTNG.TNGsimulation import Snapshot

from pynbody.array import SimArray

from .import GalaxyPoseTrajectory
from .pynbody_decorate import make_star_birth as _make_star_birth
from .pynbody_decorate import StarBirth, Unit



def units_transform(array: SimArray, new_units: Union[str, Unit], **convertion_context: Any) -> SimArray:
    """
    Convert a `pynbody` SimArray to new units and return a new SimArray.

    Parameters
    ----------
    array : pynbody.array.SimArray
        Input array with units.
    new_units : str or pynbody.units.Unit
        Target units.
    **convertion_context
        Conversion context passed to ``array.units.ratio(...)`` (e.g. ``a=...``, ``h=...``).

    Returns
    -------
    pynbody.array.SimArray
        A new array in `new_units` (input is not modified).
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
    Construct :class:`galpos.pynbody_decorate.StarBirth` for one (sub)halo in AnastrisTNG.

    The function loads star particle birth properties and host galaxy evolution,
    builds a :class:`galpos.GalaxyPoseTrajectory`, and returns a :class:`StarBirth`
    snapshot.

    Parameters
    ----------
    snapshot : AnastrisTNG.TNGsimulation.Snapshot
        Simulation snapshot handle.
    ID : int
        Subhalo/Halo identifier.
    issubhalo : bool, default=True
        If True, treat `ID` as a Subhalo; otherwise a Halo.
    host_t, host_pos, host_vel : SimArray, optional
        User-provided host trajectory samples. If all provided, host evolution
        will not be loaded from the snapshot for pose.
    angular_momentum : ndarray or SimArray, optional
        User-provided host angular momentum samples used for orientation.
    useCM : bool, default=False
        If True, use center-of-mass positions when available.
    useBirthmass : bool, default=False
        If True, use birth mass (GFM_InitialMass); else use current mass.

    Returns
    -------
    StarBirth
        Birth snapshot decorated with cosmology properties.

    Examples
    --------
    >>> # sb = make_star_birth(snapshot, ID=..., issubhalo=True)
    >>> # sb = sb.align_with_galaxy()
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

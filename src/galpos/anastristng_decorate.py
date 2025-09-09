from typing import Optional, Union

import numpy as np
from AnastrisTNG.TNGsimulation import Snapshot

from pynbody.array import SimArray

from .import GalaxyPoseTrajectory
from .pynbody_decorate import make_star_birth as _make_star_birth
from .pynbody_decorate import StarBirth





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
    originfield = snapshot.load_particle_para['star_fields'].copy()
    snapshot.load_particle_para['star_fields'] = [
        'Coordinates', 'Velocities', 'Masses', 'ParticleIDs',
        'GFM_StellarFormationTime', 'GFM_InitialMass', 'BirthPos', 'BirthVel']
    user_pose = False
    user_orient = False
    if host_t is not None:
        if host_pos is not None and host_vel is not None:
            user_pose = True
        if angular_momentum is not None:
            user_orient = True

    
    
    if issubhalo:
        PT = snapshot.load_particle(
            ID, groupType = 'Subhalo', decorate = False, order = 'star',
            )
        evo = snapshot.galaxy_evolution(
            ID, 
            ['SubhaloPos', 'SubhaloVel', 'SubhaloSpin','SubhaloCM'], 
            physical_units=False
        )
        if user_pose:
            times = host_t
            pos = host_pos
            vel = host_vel
        else:
            times = evo['t']
            pos = evo['SubhaloPos'] if not useCM else evo['SubhaloCM']
            vel = evo['SubhaloVel']

        if user_orient:
            orientation_times = host_t
            ang_mom = angular_momentum
        else:
            orientation_times = evo['t']
            ang_mom = evo['SubhaloSpin']
    else:
        
        PT = snapshot.load_particle(
            ID, groupType = 'Halo',decorate = False, order = 'star',
            )
        evo = snapshot.halo_evolution(
            ID, physical_units=False
        )
        
        if user_pose:
            times = host_t
            pos = host_pos
            vel = host_vel
        else:
            times = evo['t']
            pos = evo['GroupPos'] if not useCM else evo['GroupCM']
            vel = evo['GroupVel']

        if user_orient:
            orientation_times = host_t
            ang_mom = angular_momentum
        else:
            orientation_times = None
            ang_mom = None

    orbit = GalaxyPoseTrajectory(
        times, pos, vel, 
        angular_momentum=ang_mom, orientation_times=orientation_times)
    
    birth_time =  PT['tform']
    birth_pos = PT['BirthPos']
    birth_velocity = PT['BirthVel']
    mass = PT['mass'] if not useBirthmass else PT['GFM_InitialMass']
    scale_factor = PT['aform']
    
    snapshot.load_particle_para['star_fields'] = originfield
    
    return _make_star_birth(
        orbit, birth_time, birth_pos, birth_velocity, mass, scale_factor,
        birth_pos_units=birth_pos.units, birth_time_units=birth_time.units, 
        birth_velocity_units=birth_velocity.units, mass_units=mass.units)


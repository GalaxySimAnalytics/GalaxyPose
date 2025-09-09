""" 
This script saves galaxy position data from the TNG50 simulation to an HDF5 file.

Utilizing the AnastrisTNG package, https://github.com/wx-ys/AnastrisTNG

IDall can be changed to specify the galaxy IDs to track.

"""

from AnastrisTNG import TNGsimulation
import numpy as np
import h5py
import os
from tqdm import tqdm
    
def save_dict_to_hdf5(file: h5py.File | h5py.Group | str, data_dict: dict | None = None) ->  h5py.File | h5py.Group | None:
    """Save a dictionary to an HDF5 file or group."""
    if data_dict is None:
        data_dict = {}
    if isinstance(file, str):
        if not os.path.exists(file):
            f = h5py.File(file, "w")
            f.close()
        with h5py.File(file, "r+") as f:
            for i in data_dict:
                f.create_dataset(i, data=data_dict[i])
    if isinstance(file, h5py.File) or isinstance(file, h5py.Group):
        for i in data_dict:
            file.create_dataset(i, data=data_dict[i])
        return file
    return None

def check_path_exists(file_path, path_to_check):
    if not os.path.exists(file_path):
        return False
    
    with h5py.File(file_path, 'r') as f:
        return path_to_check in f

data_file = "TNG50_disk_trajectory.hdf5"  # file name to save galaxy positions

run = 'TNG50' 
BasePath ='/home/tnguser/sims.TNG/'+run + '-1/output'
snap =99
snapshot = TNGsimulation.Snapshot(BasePath,snap)


with h5py.File("tng50-1_bar_size.hdf5","r") as disk_file:
    ID_disk = disk_file["gal_ID"][...]

IDall = ID_disk # all snap99 galaxy ID to track

for i in tqdm(IDall):
    ID = i
    galaxy_evo = snapshot.galaxy_evolution(ID)
    
    if 'SnapNum' not in galaxy_evo or len(galaxy_evo['SnapNum']) == 0:
        print(f"Warning: No evolution data for galaxy ID {ID}, skipping...")
        continue

    for j in galaxy_evo['SnapNum']:

        snap = j
        
        path_to_check = f"{ID}/{j}"
        if check_path_exists(data_file, path_to_check):
            print(f"Data for galaxy ID {ID}, snap {j} already exists. Skipping...")
            continue
        
        match_indices = galaxy_evo['SnapNum'] == snap
        if not np.any(match_indices):
            print(f"Warning: No match for snapshot {snap} in galaxy {ID}, skipping...")
            continue
        
        subfind_id = int(galaxy_evo['SubfindID'][match_indices][0])
        
        try:
            snapshot_i = TNGsimulation.Snapshot(BasePath,snap)
            
            sub_i=snapshot_i.load_particle(subfind_id)

            if len(sub_i.s)>=100:
                alignwith = "star"
            elif (len(sub_i.g) + len(sub_i.s))>=100:
                alignwith = "baryon"
            else:
                alignwith = "all"

            sub_i.physical_units()
            sub_i.check_boundary()
            mode = 'ssc'
            
            pos_center = sub_i.center(mode=mode)
            sub_i.shift(pos=pos_center)

            re = sub_i.r(0.5,calfor=alignwith)

            vel_center = sub_i.vel_center(pos=[0,0,0.],r_cal=0.5*re)
            sub_i.shift(vel=vel_center)
            
            ang_mom = sub_i.ang_mom_vec(alignwith=alignwith,rmax = 2*re)
            
            
            a = sub_i.properties['a']
            t = sub_i.properties['t'].in_units("Gyr")
            
            galaxy_pos = pos_center.in_units("a kpc",a = sub_i.properties['a'])
            galaxy_vel = vel_center.in_units("a kpc Gyr**-1",a = sub_i.properties['a'])
            
            
            
            
            galaxy_coor = {
                f"{ID}/{j}/a": a,
                f"{ID}/{j}/t": t,
                f"{ID}/{j}/align": alignwith,
                f"{ID}/{j}/re": re,
                f"{ID}/{j}/pos": pos_center,
                f"{ID}/{j}/vel": vel_center,
                f"{ID}/{j}/cpos": galaxy_pos,
                f"{ID}/{j}/cvel": galaxy_vel,
                f"{ID}/{j}/ang_mom": ang_mom,
            }

            save_dict_to_hdf5(data_file,galaxy_coor)
            #print(f"Finished processing galaxy ID {ID}, snap: {snap}")
        except Exception as e:
            print(f"Error processing galaxy ID {ID}, snap {snap}: {str(e)}")
            continue

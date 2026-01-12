
import os
import pathlib

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

pynbody = pytest.importorskip("pynbody")
pytest.importorskip("matplotlib")

from pynbody.array import SimArray
import matplotlib.pyplot as plt
import h5py

from galpos import GalaxyPoseTrajectory
from galpos.decorate import make_star_birth

ID = 636729
test_data_path = pathlib.Path(__file__).parent.parent / "testdata"
test_data = test_data_path / f"TNG50_disk_star_Subfind_{ID}.hdf5"
cosmology_parameter = ("h","omegaM0","omegaL0","omegaB0","sigma8","ns")



def _artifact_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    If GALPOS_TEST_ARTIFACTS_DIR is set, write plots there (for CI upload-artifact).
    Otherwise write into testdata/_artifacts by default.
    """
    d = os.environ.get("GALPOS_TEST_ARTIFACTS_DIR")
    if d:
        out = pathlib.Path(d)
    else:
        out = test_data_path / "_artifacts"

    out.mkdir(parents=True, exist_ok=True)
    return out


def get_cur_sub():
    """Load test data using pynbody."""
    with h5py.File(test_data, "r") as file:
        sub = pynbody.new(star=len(file[f"{ID}/mass"]))
        sub["pos"] = file[f"{ID}/pos"][...]
        sub["vel"] = file[f"{ID}/vel"][...]
        sub["mass"] = file[f"{ID}/mass"][...]
        sub["pos"].units = "kpc"
        sub["vel"].units = "km s**-1"
        sub["mass"].units = "Msol"
        sub["aform"] = file[f"{ID}/aform"][...]
        sub.properties.update(boxsize=file["box_size"][...])
        sub.properties.update({i: file[f"{i}"][...] for i in cosmology_parameter})
        sub.properties["time"] = SimArray(13.802718326812732, "Gyr")  # in Gyr
        sub.properties["a"] = 1.0
    return sub

def get_birth(sub=None):
    """Load trajectory data using h5py."""
    sub = sub if sub is not None else get_cur_sub()
    with h5py.File(test_data, "r") as file:
        ang = file["trajectory/ang_mom"][...]  # in kpc km/s
        cpos = file["trajectory/cpos"][...]  # in a kpc
        cvel = file["trajectory/cvel"][...]  # in a kpc / Gyr
        t = file["trajectory/t"][...]  # in Gyr
        boxsize = file["box_size"][...]

        birth_mass = file[f"{ID}/birth_mass"][...]  # in Msol
        birth_pos = file[f"{ID}/birth_pos"][...]  # in a kpc
        birth_vel = file[f"{ID}/birth_vel"][...]  # in kpc Gyr**-1 a**1/2

    traj = GalaxyPoseTrajectory(
        times=t,
        positions=cpos,
        velocities=cvel,
        angular_momentum=ang,
        box_size=boxsize,
    )
    birth = make_star_birth(
        galaxy_orbit=traj,
        birth_time=sub["tform"],
        birth_pos=birth_pos,
        birth_velocity=birth_vel,
        mass=birth_mass,
        scale_factor=sub["aform"],
        birth_pos_units="a kpc",
        birth_velocity_units="kpc Gyr**-1 a**1/2",
        cosmology_params={i: sub.properties[i] for i in cosmology_parameter},
    )
    return birth

@pytest.mark.pynbody
def test_pynbody_decorate_workflow(tmp_path):
    sub = get_cur_sub()
    birth_1 = get_birth()
    birth_2 = get_birth()

    outdir = _artifact_dir(tmp_path)

    fig = birth_1.galaxy_orbit.plot(show_sampling=True)
    plt.savefig(outdir / f"Subfind_{ID}_orbit.png", dpi=150, bbox_inches="tight")

    fig = birth_1.galaxy_orbit.plot3d()
    plt.savefig(outdir / f"Subfind_{ID}_orbit3d.png", dpi=150, bbox_inches="tight")

    from galpos.plot import plot_sfr_evolution

    birth_1.align_with_galaxy(orientation_align=False)
    birth_2.align_with_galaxy(orientation_align=True)

    fig = plot_sfr_evolution(sub, birth_1, birth_2)
    plt.savefig(outdir / f"Subfind_{ID}_sfr_evolution.png", dpi=150, bbox_inches="tight")

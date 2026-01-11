# GalaxyPose

GalaxyPose is a Python toolkit for analyzing cosmological simulation data. It provides continuous models for **galaxy trajectories (position/velocity)** and **orientations** from discrete simulation snapshots, enabling galaxy states to be evaluated at arbitrary times.

[中文版本](./README_cn.md)

# Overview
Cosmological simulations output snapshots containing galaxy properties such as position, velocity, and angular momentum. Because snapshots are separated by finite time intervals, many analyses require a continuous description of a galaxy’s motion and orientation between snapshots. GalaxyPose offers interpolation-based models to bridge these gaps and query galaxy states at any time within (and optionally beyond) the sampled range.

## Key Features

- **Trajectory interpolation** with periodic-box support (unwrap/wrap handling).
- **Orientation interpolation**
  - from rotation matrices (quaternion-based smooth interpolation), or
  - from angular momentum directions (useful when only disk axis is needed).
- **Birth-frame alignment utilities** to express stellar birth positions/velocities in the host-galaxy frame (via `pynbody`).

# Installation

```bash
git clone https://github.com/GalaxySimAnalytics/GalaxyPose.git
cd GalaxyPose
pip install -e .
```

## Optional dependencies

### Plotting

Plotting helpers (e.g. `GalaxyPoseTrajectory.plot()` / `plot3d()`) require `matplotlib`.

- `pip install matplotlib`
- or `pip install -e .[plot]`

### Decorate integrations

Birth-frame alignment helpers under `galpos.decorate` are optional.

- Pynbody integration: `pip install pynbody` or `pip install -e .[decorate]`
- AnastrisTNG integration: `pip install -e .[decorate-tng]`

## Use Cases

In cosmological hydrodynamic simulations, stellar formation properties (formation time, birth position, birth velocity) are often recorded in the simulation box frame. To compute quantities relative to a host galaxy at formation time, you need the host galaxy’s position, velocity, and (optionally) orientation at that same moment. GalaxyPose supports building these continuous models and aligning particle birth properties to the host-galaxy frame.

[![sfr_evolution](./examples/sfr_evolution.png)](./examples/sfr_evolution.png)
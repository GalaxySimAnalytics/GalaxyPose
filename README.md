# GalaxyPose

GalaxyPose is a Python toolkit for analyzing cosmological simulation data. It focuses on building galaxy trajectory and orientation evolution models, addressing the issue of time intervals between cosmological simulation snapshots, and enabling researchers to obtain galaxy state information at any arbitrary moment.


[中文版本](https://github.com/GalaxySimAnalytics/GalaxyPose/blob/main/README_cn.md)

# Overview
In cosmological simulations, output snapshots contain physical quantities such as galaxy position, velocity, and angular momentum. However, due to time intervals between these snapshots, we need to construct continuous evolution models to infer galaxy states at arbitrary times. GalaxyPose is designed to solve this problem, and is also applicable for processing other similar trajectory and orientation data.

# Installation

```bash
git clone https://github.com/GalaxySimAnalytics/GalaxyPose.git
cd Galyst
pip install -e .
```

# Use Cases

In cosmological hydrodynamic simulations, we can extract information about galaxy positions, velocities, and angular momenta from different time snapshots, as well as stellar formation times, velocities, and positions. Since stellar formation information is typically recorded relative to the simulation box coordinate system, researchers need to build galaxy trajectory and orientation evolution models to determine galaxy states at any given moment, thereby calculating positional information relative to the host galaxy at the time of star formation.

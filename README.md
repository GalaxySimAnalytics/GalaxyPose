# GalaxyPose
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18212506.svg)](https://doi.org/10.5281/zenodo.18212506)

GalaxyPose is a Python toolkit for modeling galaxy **trajectories (position/velocity)** and **orientations** as continuous functions of time from discrete cosmological simulation snapshots.

[中文版本](./README_cn.md)

## Features
- Trajectory interpolation with periodic-box support (unwrap/wrap).
- Orientation interpolation
  - from rotation matrices (smooth quaternion-based interpolation), or
  - from angular-momentum directions (disk-axis use cases).
- Optional birth-frame alignment utilities via `pynbody` (`galpos.decorate`).
- Optional helpers for **IllustrisTNG (TNG simulation)** catalogs via `AnastrisTNG` (`galpos.decorate`).

## Installation

From source:
```bash
git clone https://github.com/GalaxySimAnalytics/GalaxyPose.git
cd GalaxyPose
pip install -e .
```

Optional extras:
```bash
pip install -e ".[plot]"      # matplotlib plotting helpers
pip install -e ".[decorate]"  # pynbody integration
pip install -e ".[decorate-tng]" # IllustrisTNG helpers via AnastrisTNG
```

## Use Cases

In cosmological hydrodynamic simulations, stellar formation properties (formation time, birth position, birth velocity) are often recorded in the simulation box frame. To compute quantities relative to a host galaxy at formation time, you need the host galaxy’s position, velocity, and (optionally) orientation at that same moment. GalaxyPose supports building these continuous models and aligning particle birth properties to the host-galaxy frame.

[![sfr_evolution](./examples/sfr_evolution.png)](./examples/sfr_evolution.png)


## Citation / Acknowledging GalaxyPose
If you use GalaxyPose in research, please cite the Zenodo record:

- DOI: https://doi.org/10.5281/zenodo.18212506

BibTeX:
```bibtex
@software{Lu2026GalaxyPose,
  author       = {Lu, Shuai},
  title        = {GalaxySimAnalytics/GalaxyPose: v1.0.0},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.18212506},
  url          = {https://doi.org/10.5281/zenodo.18212506},
}
```


You can also use the metadata in [`CITATION.cff`](./CITATION.cff).

## License
MIT License. See [`LICENSE`](./LICENSE).

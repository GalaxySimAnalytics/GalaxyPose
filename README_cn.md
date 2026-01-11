[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18212505.svg)](https://doi.org/10.5281/zenodo.18212505)
[![codecov](https://codecov.io/gh/GalaxySimAnalytics/GalaxyPose/branch/main/graph/badge.svg)](https://codecov.io/gh/GalaxySimAnalytics/GalaxyPose)
# GalaxyPose

GalaxyPose 是一个用于宇宙学模拟数据分析的 Python 工具包，可将离散快照中的**星系轨迹（位置/速度）**与**姿态（朝向）**构建为连续模型，从而在任意时刻评估星系状态。

## 主要特性
- 轨迹插值，支持周期盒（unwrap/wrap）处理。
- 姿态插值：
  - 基于旋转矩阵（四元数平滑插值），或
  - 基于角动量方向（适用于只关心盘面法向的场景）。
- 可选：`galpos.decorate`（依赖 `pynbody`）用于恒星诞生坐标对齐。
- 可选：面向 **IllustrisTNG（TNG 模拟）** 数据目录/目录结构的辅助功能（依赖 `AnastrisTNG`，见 `galpos.decorate`）。


## 安装

从源码安装：
```bash
git clone https://github.com/GalaxySimAnalytics/GalaxyPose.git
cd GalaxyPose
pip install -e .
```

可选依赖：
```bash
pip install -e ".[plot]"        # 绘图功能（matplotlib）
pip install -e ".[decorate]"    # pynbody 集成
pip install "AnastrisTNG @ git+https://github.com/wx-ys/AnastrisTNG" # IllustrisTNG 辅助功能（AnastrisTNG）
```

## 应用场景

在宇宙学流体模拟中，恒星形成时间、诞生位置与诞生速度通常记录在模拟盒坐标系下。若要计算“相对于宿主星系”的形成环境，就需要在恒星形成时刻得到宿主星系的位置、速度以及（可选的）姿态。GalaxyPose 用于构建这些连续模型，并可将粒子的诞生信息对齐到宿主星系参考系中。

[![sfr_evolution](./examples/sfr_evolution.png)](./examples/sfr_evolution.png)

## 引用 / 致谢（Acknowledging the code）
如果你在论文或报告中使用了 GalaxyPose，请引用 Zenodo 记录：

- DOI: https://doi.org/10.5281/zenodo.18212505


BibTeX:
```bibtex
@software{Lu2026GalaxyPose,
  author       = {Lu, Shuai},
  title        = {GalaxyPose},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18212505},
  url          = {https://doi.org/10.5281/zenodo.18212505},
}
```

也可以直接使用仓库中的 [`CITATION.cff`](./CITATION.cff) 进行引用信息导出。

## 许可证
MIT License，见 [`LICENSE`](./LICENSE)。

# GalaxyPose

GalaxyPose 是一个用于宇宙学模拟数据分析的 Python 工具包。它专注于构建星系轨迹和姿态演化模型，解决宇宙学模拟快照数据之间的时间间隔问题，能够获取任意时刻的星系状态信息。


# 功能概述
在宇宙学模拟中，输出的快照包含星系的位置、速度和角动量等物理量。然而，由于这些快照之间存在时间间隔，我们需要构建连续的演化模型来推断任意时刻的星系状态。GalaxyPose 正是为解决这一问题而设计的工具包，同时也适用于处理其他类似的轨迹与姿态数据。

# 安装方法

```bash
git clone https://github.com/GalaxySimAnalytics/GalaxyPose.git
cd Galyst
pip install -e .
```

# 应用场景

在宇宙学流体模拟中，我们可以从不同时间快照中提取星系的位置、速度和角动量等信息，以及恒星的形成时间、速度和位置等数据。由于恒星形成信息通常是相对于模拟盒子坐标系记录的，研究人员需要构建星系的轨迹和姿态演化模型，以便在任意时刻确定星系的状态，进而计算恒星形成时相对于所属星系的位置信息。
[![sfr_evolution](./examples/sfr_evolution.png)](./examples/sfr_evolution.png)

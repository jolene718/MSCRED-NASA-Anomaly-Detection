# MSCRED on NASA SMAP/MSL

本项目基于 PyTorch 将经典 MSCRED 时序异常检测模型完整适配到 NASA SMAP/MSL 卫星遥测异常检测数据集，在保留原模型多尺度卷积循环架构思路的前提下，对数据预处理、动态多尺度相关矩阵生成、训练评估与可视化流程进行了全面重构，移除了原有代码中针对固定数据集的硬编码设置，修复了 ConvLSTM 时间维度与批次维度混淆的实现问题，使模型能够灵活适配 25 维 SMAP 与 55 维 MSL 两类多变量时序数据。

## 项目亮点

- 面向 NASA SMAP/MSL 数据集完成了端到端适配
- 支持混合处理 `25` 维 SMAP 通道和 `55` 维 MSL 通道
- 动态生成多尺度 signature matrices，不依赖旧示例缓存
- 修复了 ConvLSTM 时间维处理和固定 `30x30` 输入假设等实现问题
- 提供预处理、训练、评估、可视化的完整脚本

## 核心工作

### 1. 数据适配

- 解析 `labeled_anomalies.csv`
- 合并重复通道的异常区间
- 仅使用训练集拟合归一化参数
- 为每个通道构建规范化缓存，供训练和评估使用

### 2. 模型与实现修复

- 重写 ConvLSTM 输入维度逻辑，统一张量格式
- 去除固定 `30x30` 传感器矩阵假设
- 让模型能够适配可变传感器数量输入
- 保留 MSCRED 的核心思路：多尺度相关矩阵、CNN 编解码、ConvLSTM 时间建模、基于重建误差的异常检测

### 3. 训练与评估流程补全

- 增加训练集、验证集、测试集完整流程
- 支持 early stopping 和 checkpoint 保存
- 加入 per-channel threshold、top-k residual score、平滑与 point-adjusted 评估
- 支持输出 history、metrics、scores 和分通道图表

## 数据集组织方式

项目默认读取以下结构的 NASA SMAP/MSL 数据：

```text
archive/
├─ labeled_anomalies.csv
└─ data/
   └─ data/
      ├─ train/*.npy
      └─ test/*.npy
```

当前仓库中只保留轻量的标签文件：

- `archive/labeled_anomalies.csv`

运行项目所需的原始 NASA `.npy` 数据并没有包含在仓库中，需要你自行下载后放入 `archive/data/data/train/` 和 `archive/data/data/test/`。

运行过程中生成的缓存、checkpoint 和输出结果也不会预先放在仓库里。

## 快速开始

### 1. 创建环境

```powershell
conda create -n nasa-mscred python=3.10 -y
conda activate nasa-mscred
pip install -r requirements.txt
```

如果使用 CUDA 11.8：

```powershell
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 准备数据缓存

在执行这一步之前，请先确认你已经把 NASA 原始 `train/test .npy` 文件放到了 `archive/data/data/` 目录下。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_nasa_data.ps1
```

等价手动命令：

```powershell
python .\utils\matrix_generator.py --raw-data-dir .\archive\data\data --labels-path .\archive\labeled_anomalies.csv --output-dir .\data\nasa_processed --overwrite
```

### 3. 训练模型

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_nasa_release.ps1
```

### 4. 评估模型

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_nasa_release.ps1
```

## 当前结果

默认 release 后处理配置下的结果为：

- raw precision: `0.3160`
- raw recall: `0.3469`
- raw F1: `0.3307`
- adjusted precision: `0.4895`
- adjusted recall: `0.7201`
- adjusted F1: `0.5828`

项目已经完整跑通了 NASA 数据适配、训练和评估流程，后续仍有优化空间。

## 项目结构

```text
main.py
model/
utils/
scripts/
archive/
data/
requirements.txt
README.md
```

说明：

- `archive/` 用于放置标签文件和本地下载的 NASA 原始数据
- `data/` 用于放置运行时生成的缓存
- `checkpoints/` 和 `outputs/` 会在训练和评估后自动生成

关键文件：

- [main.py](./main.py)：训练入口
- [model/mscred_nasa.py](./model/mscred_nasa.py)：NASA 适配版 MSCRED
- [model/convolution_lstm.py](./model/convolution_lstm.py)：ConvLSTM 实现
- [utils/nasa.py](./utils/nasa.py)：NASA 数据解析与缓存构建
- [utils/data.py](./utils/data.py)：数据集与 signature matrix 生成
- [utils/evaluate.py](./utils/evaluate.py)：评估入口
- [utils/pipeline.py](./utils/pipeline.py)：训练、打分、指标与可视化逻辑


## 后续可以继续优化的方向

- 将 SMAP 和 MSL 分开训练，而不是单模型混训
- 继续搜索窗口组合，如 `5,20,40` 或 `20,50,100`
- 对比 Min-Max、Standardization、Robust Scaling 等归一化方案
- 引入更稳的训练策略，如更低学习率、调度器、weight decay
- 将阈值搜索整理成正式脚本

## 参考资料

- MSCRED 论文: [https://arxiv.org/abs/1811.08055](https://arxiv.org/abs/1811.08055)
- NASA SMAP/MSL 数据集页面: [https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/code](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/code)
- Telemanom: [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom)

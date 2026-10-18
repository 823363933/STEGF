# 面向动态三维高斯溅射的共享外观脚手架

[English](README.md) | 简体中文

## 项目简介

动态三维高斯溅射需要同时表达跨时间稳定的场景外观和随时间变化的动态细节。仅由逐高斯特征承担外观表示时，重复观测中的稳定证据无法通过显式空间结构共享；运动与时间不透明度独立优化时，运动复杂度也可能与高斯的可见时间范围不匹配。

本项目引入多分辨率特征网格作为**共享外观脚手架**。每个高斯在规范位置查询网格特征，并将其与自身的局部外观特征融合。分离梯度的运动—生命周期耦合根据时间支持范围调节二阶和三阶运动：短生命周期高斯趋近线性运动，长生命周期高斯保留高阶轨迹。训练完成后，脚手架贡献可烘焙至逐高斯特征，紧凑推理模型不再需要网格及其查询计算。

当前维护的模型版本是采用 `coupled_detached` 耦合的
`S2.0.4-couptest-poly`。为兼容已有实验，环境、配置、检查点和日志仍保留旧的 `STEGF` 标识。

## 4D 重建示例

![N3D cut_roasted_beef：固定视角播放、暂停时间并移动视角、移动视角并播放](assets/cut_roasted_beef_interactive_4d.gif)

依次展示固定视角播放、暂停时间并移动视角，以及移动视角并继续播放。300 帧序列由六个独立的 50 帧模型组成，每段训练 30,000 步，所有相机视角均参与训练，仅用于展示而非留出视角评测。这是预渲染演示，并非实时交互查看器；未进行额外颜色校准或分段混合。

### 主要特性

- 在高斯规范位置查询并共享多分辨率空间特征；
- 保留逐高斯外观残差以表达局部和动态细节；
- 使线性、二阶和三阶运动适配时间支持范围；
- 使用全时刻稀疏点，可选融合第 0 帧稠密点；
- 通过场景配置控制普通增密、多视角增密及 VAD/TAT；
- 将 static grid 烘焙为更小、更快的紧凑推理模型；
- 为 N3D、Technicolor 和 Google Immersive 提供统一的预处理、训练、导出与测试流程。

## 环境配置

请在仓库根目录执行命令。训练需要 Conda、NVIDIA GPU，以及与
PyTorch 2.0.0 和 CUDA 11.8 兼容的 CUDA 编译器。

```bash
bash script/setup.sh
conda activate STEGF
```

该脚本会安装训练环境，并编译本地光栅化、simple-KNN 和 MMCV KNN 扩展。

数据预处理使用独立环境：

```bash
# 本地 CUDA 12.8 / Python 3.10 配置
bash script/setup_preprocess.sh

# 服务器 CUDA 11.8 / Python 3.8 配置
bash script/setup_preprocess_server.sh

conda activate STEGF-preprocess
```

两种配置都会安装 COLMAP 4.1.1、Open3D 和图像处理依赖。默认还会安装
MiDaS 及其权重；如果场景配置不使用 MiDaS 先验，可在环境脚本后添加
`--without-midas`。无显示器服务器需安装 `xvfb` 和 `xauth`；当 COLMAP
需要显示环境时，预处理脚本会自动通过 `xvfb-run` 运行。

## 数据预处理

三个数据集采用统一的场景目录。`<datapath>` 默认是
`/root/autodl-tmp`。

```text
<datapath>/
└── <scene>/
    ├── colmap_0/
    ├── colmap_1/
    ├── ...
    ├── colmap_<duration-1>/
    ├── dense_points.ply                 # 可选的第 0 帧稠密初始化
    └── midas_beit_large_512/            # 可选的深度类先验
```

每个 `colmap_<time>` 均包含同步图像和位于 `sparse/0` 的 COLMAP 模型，完整时间序列不可缺少。只有场景配置启用对应 `field_*` 选项时，才需要可选数据。

| 数据集 | 原始数据必须提供的标定 | 预处理脚本 | 配置目录 |
| --- | --- | --- | --- |
| N3D / DyNeRF | 同步的 `cam*.mp4`、`poses_bounds.npy` | `preprocess_n3d_scene.py` | `configs/n3d_ours` |
| Technicolor | 官方 PNG 序列、`cameras_parameters.txt` | `preprocess_tech_scene.py` | `configs/tech_ours` |
| Google Immersive | 已标定的相机 MP4、`models.json` | `preprocess_immersive_scene.py` | `configs/immersive_ours` |

这些脚本使用数据集已有的相机标定，不会从任意未标定视频中估计相机位姿。默认处理 50 帧，并根据场景配置决定是否生成稠密点云和 MiDaS 数据；可用 `--dense on|off` 和 `--midas on|off` 覆盖可选阶段。

### N3D / DyNeRF

```bash
conda activate STEGF-preprocess
python script/preprocess_n3d_scene.py \
  --scene coffee_martini \
  --inputpath dataset/DyNeRF \
  --datapath /root/autodl-tmp
```

支持 `coffee_martini`、`cook_spinach`、`cut_roasted_beef`、
`flame_salmon_1`、`flame_steak` 和 `sear_steak`。处理后的场景还包含
`poses_bounds.npy`。

### Technicolor

```bash
conda activate STEGF-preprocess
python script/preprocess_tech_scene.py \
  --scene Fabien \
  --inputpath dataset/Tech \
  --datapath dataset/Tech-pre
```

支持 `Birthday`、`Fabien`、`Painter`、`Theater` 和 `Train`。脚本会根据场景配置自动选择官方序列起始位置。

### Google Immersive

```bash
conda activate STEGF-preprocess
python script/preprocess_immersive_scene.py \
  --scene 12_Cave \
  --inputpath dataset/Immersive \
  --datapath dataset/Immersive-pre
```

支持 `01_Welder`、`02_Flames`、`04_Truck`、`09_Alexa`、`10_Alexa`、
`11_Alexa` 和 `12_Cave`。Immersive 场景还会保留原始鱼眼监督帧和每个相机对应的固定采样网格：

```text
<datapath>/<scene>/
├── models.json
├── camera_0001/<time>.png
├── camera_0001.npy
├── ...
└── colmap_<time>/
```

`colmap_<time>/images` 中的针孔图像用于 COLMAP 和高斯光栅化；
`camera_*` 图像才是实际的鱼眼监督，NPY 网格负责把针孔渲染映射回鱼眼域。

## 训练与测试

`script/run_n3d_train_test.py` 是三个数据集共用的运行入口。一次标准运行依次完成：

1. 训练完整模型；
2. 测试指定的完整检查点；
3. 将共享脚手架烘焙为紧凑推理模型；
4. 测试最终紧凑检查点。

最终模型指标以紧凑推理模型为准。

### N3D

```bash
python script/run_n3d_train_test.py --scene coffee_martini
```

### Technicolor

```bash
python script/run_n3d_train_test.py \
  --dataset tech \
  --scene Fabien
```

### Google Immersive

```bash
python script/run_n3d_train_test.py \
  --dataset immersive \
  --scene 12_Cave \
  --datapath /root/autodl-tmp/Immersive \
  --savepath /root/autodl-tmp/output/Immersive
```

指定自定义根目录时，脚本读取 `<datapath>/<scene>/colmap_0`，并写入
`<savepath>/<scene>`：

```bash
python script/run_n3d_train_test.py \
  --scene cook_spinach \
  --datapath /data/N3D \
  --savepath /data/output/N3D
```

通过 `--scenes` 可以顺序运行多个场景：

```bash
python script/run_n3d_train_test.py \
  --scenes cook_spinach cut_roasted_beef flame_steak
```

常用检查和复用方式：

```bash
# 只检查路径并打印命令
python script/run_n3d_train_test.py --scene coffee_martini --dry_run

# 短 CUDA 冒烟实验
python script/run_n3d_train_test.py \
  --scene coffee_martini \
  --iterations 10 \
  --skip_test_stage \
  --skip_compact_stage

# 基于已有完整检查点导出并测试紧凑模型
python script/run_n3d_train_test.py \
  --scene coffee_martini \
  --skip_train_stage
```

默认情况下，N3D 和 Technicolor 场景从 `/root/autodl-tmp` 解析，也可始终显式指定其他数据根目录。N3D 使用 `colmapvalid`，Technicolor 使用
`technicolorvalid`，Immersive 使用 `immersivevalidss`。

## 紧凑推理模型

标准运行会自动完成紧凑导出。导出器在高斯规范位置查询训练后的脚手架，将训练相机平均贡献烘焙至逐高斯外观特征，删除仅训练时需要的网格与路由参数，并写入：

```text
<savepath>/<scene>/compact/
```

原始检查点不会被修改。也可以单独执行导出：

```bash
python script/export_compact_model.py \
  --quiet --eval \
  --test_iteration 30000 \
  --valloader colmapvalid \
  --configpath configs/n3d_ours/coffee_martini.json \
  --model_path /root/autodl-tmp/output/coffee_martini \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0
```

导出器会在内存中验证代表帧，并在紧凑模型目录保存
`compact_config.json`。如需继续训练，应从原始完整检查点恢复，而不是紧凑模型。

## 4D 轨迹视频

标准测试在时间推进时保持测试相机不动。定性展示 4D 结果时，
`render_4d_video.py` 会连续推进时间，并根据真实标定相机阵列的尺度生成一条平滑的虚拟相机轨迹。建议直接渲染紧凑模型：

```bash
python script/render_4d_video.py \
  --quiet --eval \
  --model_path /root/autodl-tmp/output/cut_roasted_beef/compact \
  --source_path /root/autodl-tmp/cut_roasted_beef/colmap_0 \
  --test_iteration 30000 \
  --valloader colmapvalid \
  --frames 150 \
  --fps 30
```

默认输出为 `<model_path>/trajectory_4d/trajectory_4d.mp4`，同时保存封面帧和记录相机位姿、时间戳的 JSON 文件。`--trajectory_scale` 根据真实相机分布控制视角运动幅度，`--output_scale` 在保持视场角不变的情况下调整输出分辨率。该轨迹限制在标定视角邻域内，不使用旧版写死半径的 `colmapmv` 螺旋路径。

## 仓库结构

```text
train.py, test.py                         训练与测试入口
helper_train.py, helper_model.py          共享训练与模型工具
configs/{n3d,tech,immersive}_ours/        数据集和场景配置
script/run_n3d_train_test.py              统一实验入口
script/preprocess_*_scene.py              数据集预处理脚本
script/export_compact_model.py            紧凑推理模型导出
script/render_4d_video.py                  变化视角的时空轨迹渲染
thirdparty/gaussian_splatting/             模型、渲染器与 CUDA 源码
```

数据集、检查点和生成的实验输出均由 Git 忽略。

## 致谢

本项目建立在 STGS 官方实现之上。数据流程还参考了 STGS 的预处理与
Immersive 鱼眼处理、E-D3DGS 的第 0 帧稠密重建与下采样方式，以及 MiDaS
的 `dpt_beit_large_512` 模型。请遵守各上游项目的许可证。

```bibtex
@inproceedings{li2024stgs,
  title={Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
  author={Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
  booktitle={CVPR},
  year={2024}
}
```

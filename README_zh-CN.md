# 面向动态三维高斯溅射的共享外观脚手架

[English](README.md) | 简体中文

## 摘要

动态三维高斯溅射可以同时重建跨时间稳定的场景内容与随时间变化的动态细节。然而，仅以高斯基元为特征载体的外观表示缺少显式的空间共享结构，使重复观测中的稳定信息分散于各高斯基元，导致部分稳定信息难以被充分学习。同时，独立参数化的运动与时间不透明度函数缺少与时间支持范围相适应的优化机制，难以协调短时局部变化与长时复杂运动的建模需求，限制了动态区域的重建质量。

为此，本文在高斯基元的基础上，构建多分辨率特征网格作为脚手架，将空间共享的外观学习与时间尺度适配的运动优化相结合，提出了共享外观的优化框架。将位置函数的常数项作为高斯基元规范位置，通过规范位置查询并融合共享特征，使不同高斯基元的重复观测共同更新同一组空间参数，同时保留逐高斯基元特征以表达局部外观差异，再配合高阶运动项耦合系数，根据高斯基元的时间支持范围调节高阶运动项，协调运动表达与时间可见性的优化。

该优化框架能够充分利用动态场景中的稳定信息，并促进高斯基元对动态细节的有效表达，以紧凑的表示兼顾静态内容与动态细节的重建。此外，脚手架提供的共享外观特征可在训练结束后烘焙至高斯基元，使推理过程无需保留特征网格及其查询计算，在保持重建质量的同时减小模型存储并提升渲染速度。

> **代码名称兼容说明。** 为保证已有实验能够复现，仓库路径、Conda 环境、配置命名空间、序列化元数据以及 `[STEGF]` 日志前缀继续保留旧的 `STEGF` 标识。本文档使用的方法名称为 **面向动态三维高斯溅射的共享外观脚手架**。

## 方法概述

当前维护的研究实现为 `S2.0.4-couptest-poly`，采用 `coupled_detached` 运动—不透明度耦合。其有效设计包括：

- **高斯场景表示。** 动态三维高斯基元分别保留自身的几何、局部外观特征、运动系数和时间不透明度核。
- **共享外观脚手架。** 在固定空间坐标中建立多分辨率特征网格，并通过高斯规范位置查询网格。不同高斯的稳定、重复观测可以共同更新同一组空间参数。这里的 “static” 表示网格节点坐标固定，并非对物理静态物体进行硬语义分类。
- **局部外观残差。** 逐高斯特征始终保留，使不能由共享脚手架一致解释的局部细节仍然可以得到表达。
- **运动—生命周期耦合。** 每个高斯具有线性、二次和三次运动系数。二次与三次位移由断开梯度的归一化时间宽度进行缩放：短时间支持的高斯趋近线性运动，宽时间支持的高斯保留高阶运动能力。
- **初始化与拓扑调整。** 官方场景配置使用全时刻 COLMAP 稀疏点云与第 0 帧稠密点云联合初始化，随后执行相应场景的普通增密和多视角结构增密调度。
- **光度辅助。** 共享脚手架配合可训练的亮度/白平衡曝光修正与深度相关尺度正则化。
- **紧凑推理导出。** 训练结束后，将训练相机平均的脚手架贡献烘焙到规范位置处的逐高斯外观特征中。紧凑检查点不再保存训练期网格和路由辅助参数。
- **场景特定的 VAD/TAT。** 只有经过实验确认能够获益的场景才启用时间不透明度加权梯度积累和生命周期自适应增密阈值；其他场景保持标准调度。

当前配置已禁用被放弃的研究分支，包括动态 Euler 网格、分阶段路由、旧版 EMS、时间细化、Carrier 运动、时空泡和分段速度运动。N3D 的 `coffee_martini` 仍是主要参考配置与复现场景，但模型并未硬编码限制于该场景。对应配置为 `configs/n3d_ours/coffee_martini.json`。

## 环境配置

所有命令均应从仓库根目录运行。由于光栅化器、simple-KNN 和 MMCV KNN 算子需要在本地编译，因此需要 Conda、NVIDIA 驱动以及可用的 CUDA 编译器。

```bash
bash script/setup.sh
conda activate STEGF
```

环境脚本会创建或更新 Python 3.8 的 `STEGF` Conda 环境，从 CUDA 11.8 wheel 源安装 PyTorch 2.0.0、torchvision 0.15.0 和 torchaudio 2.0.0，编译三个本地扩展，并验证其能否导入。编译扩展时，系统中的 `nvcc` 必须与该 PyTorch 构建兼容。

视频预处理使用独立环境，因为 COLMAP、Open3D 和 MiDaS 并非训练期依赖：

```bash
bash script/setup_preprocess.sh
conda activate STEGF-preprocess
```

默认预处理配置面向本地 Python 3.10 环境，并从 CUDA 12.8 wheel 源安装 PyTorch 2.7.1。对于使用 Python 3.8、PyTorch 2.0.0 和 CUDA 11.8 训练栈的服务器，使用独立的服务器配置：

```bash
bash script/setup_preprocess_server.sh
conda activate STEGF-preprocess
```

服务器配置安装 Python 3.8、PyTorch 2.0.0、torchvision 0.15.0 和 torchaudio 2.0.0。两种配置都会安装支持 CUDA 的 COLMAP 4.1.1，以及图像、点云和深度推理所需的软件包；同时将固定版本的 MiDaS 源码和 `dpt_beit_large_512.pt` 权重放入被 Git 忽略的 `thirdparty/MiDaS/`。预处理完成后，训练不再依赖 Open3D 或 MiDaS。若配置不生成 MiDaS 先验，可在任一环境安装命令后添加 `--without-midas`，跳过对应的 PyTorch 依赖、源码和模型下载。

在无显示器的 Ubuntu 服务器上，运行场景预处理前需要安装一次虚拟显示器：

```bash
apt-get update
apt-get install -y xvfb xauth
```

当仍需运行 COLMAP 且未设置 `DISPLAY` 时，N3D、Technicolor 和 Immersive 一体化预处理脚本会自动在 `xvfb-run -a` 下重新启动。如果 COLMAP 产物已经完整，仅需执行 MiDaS 等后续阶段，则不会启动 Xvfb。

## 数据集规范

一体化运行器以相同的场景无关结构支持 N3D、Technicolor 和 Google Immersive。令 `<location>` 表示通过 `--datapath` 指定的数据根目录，`<scene>` 表示通过 `--scene` 指定的场景名称：

```text
<location>/
├── <scene>/
│   ├── colmap_0/
│   ├── colmap_1/
│   ├── ...
│   ├── colmap_<duration-1>/
│   ├── poses_bounds.npy                 # 仅 N3D 必需
│   ├── cameras_parameters.txt           # Technicolor 来源记录
│   ├── dense_points.ply                 # 可选
│   └── midas_beit_large_512/            # 可选
└── output/
    └── <scene>/
```

默认 `<location>` 为 `/root/autodl-tmp`。标准命令将 `<location>/<scene>/colmap_0` 作为 `--source_path`，加载器会自动发现同级的其余 `colmap_*` 时间切片。未显式传入 `--savepath` 时，输出写入 `<location>/output/<scene>`。

完整的 `colmap_<time>` 序列不可缺少。N3D 还要求 `poses_bounds.npy`。Technicolor 的相机标定会在预处理阶段转换为 COLMAP 模型，因此训练不依赖 `poses_bounds.npy`；复制的 `cameras_parameters.txt` 仅用于记录来源。只有 `field_dense_initialization=1` 时才要求 `dense_points.ply`。只有同时启用背景补点及其 MiDaS 过滤时才要求 `midas_beit_large_512/`。通用 N3D `default.json` 会关闭这两项可选机制，专用场景配置只启用该场景已准备好的输入。

Immersive 场景还需要原始鱼眼监督帧，以及每个标定相机的一份固定采样网格：

```text
<location>/<scene>/
├── models.json
├── camera_0001/
│   ├── 0.png
│   ├── ...
│   └── <duration-1>.png
├── camera_0001.npy
├── ...
└── colmap_<time>/
```

`colmap_<time>/images` 中的图像是用于标定三角化和高斯光栅化的宽视场针孔图像。`camera_*/<time>.png` 保留在原始鱼眼域，是实际 RGB 监督。NPY 文件通过固定的 `grid_sample` 查找，将针孔渲染结果映射回鱼眼域。因此，即使禁用可选的稠密点云和 MiDaS 阶段，这些 NPY 文件也不可缺少。

### 从视频一体化预处理 N3D

N3D 一体化预处理脚本接受一个已经标定的 N3D/DyNeRF 风格场景：

```text
<raw-location>/<scene>/
├── cam00.mp4
├── cam01.mp4
├── ...
└── poses_bounds.npy
```

视频必须同步，`poses_bounds.npy` 的行必须与按文件名排序后的 `cam*.mp4` 一一对应。脚本使用已有的 LLFF/N3D 相机标定，不会为未标定视频自行估计相机位姿。

仓库为六个公开 N3D 场景提供了配置：`coffee_martini`、`cook_spinach`、`cut_roasted_beef`、`flame_salmon_1`、`flame_steak` 和 `sear_steak`。它们以 `resolution=2` 使用连续 50 帧。除非传入 `--source_start_frame`，否则从源视频第 0 帧开始。

使用一条命令生成解析后场景配置所启用的全部产物：

```bash
conda activate STEGF-preprocess
python script/preprocess_n3d_scene.py \
  --scene <scene> \
  --inputpath <raw-location> \
  --datapath <location>
```

例如，本地 N3D 数据位于 `dataset/DyNeRF` 时：

```bash
python script/preprocess_n3d_scene.py \
  --scene coffee_martini \
  --inputpath dataset/DyNeRF \
  --datapath /root/autodl-tmp
```

处理流程依次执行：

1. 从每个相机视频提取配置规定的序列，并在每个时间索引独立执行 SpacetimeGaussians 风格的固定位姿 COLMAP 重建。
2. 若启用稠密初始化，在第 0 时刻多视角图像上执行 COLMAP PatchMatch 和融合，再应用 E-D3DGS 的迭代体素下采样规则，直到点数不超过 100,000。
3. 若启用 MiDaS 过滤的背景补点，为 `field_bg_dense_add_time_indices` 指定的时间索引生成 `dpt_beit_large_512` 深度类数组。

`--dense auto` 和 `--midas auto` 为默认设置，它们遵循解析后的配置（优先 `<scene>.json`，随后为 `default.json`）。可以通过 `--dense on|off` 和 `--midas on|off` 在不修改模型配置的情况下覆盖产物生成。其他常用参数包括 `--duration`、`--source_start_frame`、`--downscale`、`--gpu_index` 和 `--dry_run`。

已经完成的 `colmap_<time>` 会被跳过。只有指定 `--restart_incomplete` 时，脚本才会替换中断的隐藏构建工作区；已经发布但不完整的 `colmap_<time>` 不会被自动删除。

### Technicolor 预处理

Technicolor 预处理脚本读取官方无畸变 PNG 序列与 `cameras_parameters.txt`，构建相同的 50 切片 COLMAP 结构，并根据解析后的场景配置选择是否生成 `dense_points.ply`：

```bash
conda activate STEGF-preprocess
python script/preprocess_tech_scene.py \
  --scene Fabien \
  --inputpath dataset/Tech \
  --datapath dataset/Tech-pre
```

公开 Technicolor 配置包括 `Birthday`、`Fabien`、`Painter`、`Theater` 和 `Train`。除非传入 `--source_start_frame`，否则脚本会自动使用官方序列偏移。当前最终配置启用第 0 帧稠密初始化，不启用 N3D 开放背景的 MiDaS 路径。

### Google Immersive 预处理

将官方 Immersive ZIP 解压到 `dataset/Immersive`，然后执行：

```bash
conda activate STEGF-preprocess
python script/preprocess_immersive_scene.py \
  --scene 04_Truck \
  --inputpath dataset/Immersive \
  --datapath dataset/Immersive-pre
```

支持的七个规范场景名为 `01_Welder`、`02_Flames`、`04_Truck`、`09_Alexa`、`10_Alexa`、`11_Alexa` 和 `12_Cave`。脚本也接受三个 Alexa 场景的官方长目录名，并默认以规范短名称发布。脚本从 `models.json` 读取相机位姿、内参与鱼眼系数，仅解码请求的源帧，忽略没有标定记录的 MP4。

默认处理前 50 帧。中间针孔模型的焦距倍率与 STGS full 路径一致：`02_Flames` 为 `0.35`，其他场景为 `0.36`；可通过 `--focal_scale` 覆盖。后续运行会复用已完成的畸变帧、采样网格和 `colmap_<time>`。与其他预处理脚本相同，当存在 Immersive 配置时，`--dense auto` 和 `--midas auto` 遵循 `configs/immersive_ours/<scene>.json`。七个官方场景配置均启用第 0 帧稠密初始化，并关闭 N3D 特定的 MiDaS 背景补点路径。通用 `default.json` 为未配置的新场景关闭两项可选阶段。也可以通过 `--dense on|off` 或 `--midas on|off` 覆盖产物生成决定。

### 分时刻 RGB 图像与稀疏点云

对于 `duration=N` 的序列，场景必须包含从 `colmap_0` 到 `colmap_<N-1>` 的所有目录。每个时间切片至少应具有：

```text
<location>/<scene>/colmap_<time>/
├── images/
│   ├── <camera-0>.png
│   ├── <camera-1>.png
│   └── ...
└── sparse/0/
    └── points3D.bin
```

所有时间切片必须包含相同的相机图像名称。此外，`colmap_0/sparse/0/` 必须包含 `cameras.bin` 和 `images.bin`，其中的图像名称必须与 PNG 文件对应。N3D 场景还需要场景级 `poses_bounds.npy`，Technicolor 不需要。

加载器会将全部时间切片中的 `points3D.bin` 合并为 `colmap_0/sparse/0/points3D_total<N>.ply`。该文件是可选生成缓存；若不存在，则所有源 `points3D.bin` 必须可用，并且 `colmap_0/sparse/0` 必须可写。

启用 `--eval` 后，按自然文件名排序的第一个相机作为所有时间步的测试视角，其余相机作为训练视角。最终训练路径不会读取原始 COLMAP 的 `distorted`、`manual`、`stereo`、`tmp` 或 `input.db`。

### 第 0 帧稠密初始化

`<location>/<scene>/dense_points.ply` 是时间索引 0 的可选稠密点云。模型初始化时会将其与全时刻稀疏点融合；只有 `field_dense_initialization=1` 时该文件才是硬性要求。

PLY 必须包含 `x`、`y`、`z`，以及 `red`、`green`、`blue` 或 `r`、`g`、`b` 属性。预期顶点数是场景相关的：可以在场景配置中将 `field_dense_initialization_expected_points` 设为已知数量，或设为 `0` 只关闭精确数量检查。标准相对路径为 `../dense_points.ply`，以 `colmap_0` 为基准解析。97,877 仅属于历史 `coffee_martini` 产物，其他场景不需要生成完全相同的点数。

### MiDaS-BEiT 深度类数组

当 `field_bg_dense_add=1` 且 `field_bg_dense_beit_filter=1` 时，场景配置使用的每个背景增密时刻和相机都需要：

```text
<location>/<scene>/midas_beit_large_512/
└── colmap_<time>/
    ├── <camera-0>/raw_depth_like.npy
    ├── <camera-1>/raw_depth_like.npy
    └── ...
```

这些数组是 float32 深度类图。必要时，训练会将其缩放至当前渲染分辨率。训练不会读取预处理诊断文件 `band_p10_p030.png`、`stats.txt` 和 `manifest_multitime.json`。

配置启用时，一体化场景预处理脚本会自动生成这些文件，也可以单独运行：

```bash
python script/precompute_midas_beit_depth_like.py \
  --scene <scene> \
  --datapath <location> \
  --time_indices <comma-separated-time-indices>
```

独立脚本默认使用 `thirdparty/MiDaS`，将结果写入 `<location>/<scene>/midas_beit_large_512`，也可以通过 `--midas_repo` 指定其他检出目录。数组生成后，训练不再需要 MiDaS 及其权重。如果目录缺失，当前实现会发出警告并禁用过滤器，但这种运行不等价于启用该配置，表示输入不完整。

### `coffee_martini` 参考数据

提交的参考配置使用 `duration=50`，因此要求 `colmap_0` 至 `colmap_49`。每个时间切片包含以下 18 个相机图像：

```text
cam00.png  cam01.png  cam02.png  cam04.png  cam05.png  cam06.png
cam07.png  cam08.png  cam09.png  cam10.png  cam11.png  cam12.png
cam13.png  cam14.png  cam16.png  cam18.png  cam19.png  cam20.png
```

参考配置使用 900 张、分辨率为 2704×2028 的图像。启用 `--eval` 后，`cam00` 提供 50 个测试帧，其他 17 个相机提供 850 个训练视图。历史最终实验输入包含 97,877 个稠密顶点和 344,575 个全时刻稀疏点。使用相同图像重新运行 CUDA COLMAP 时，特征匹配、三角化、稠密融合和体素下采样可能产生少量差异，因此不要求逐字节复现这些点数。提交的配置将稠密精确数量检查设为 `0`；一体化预处理脚本改为验证点属性，并确保点数不超过 100,000。

最终配置仅在时间索引 0 执行 BEiT 过滤的背景增密。因此，精确复现需要在 `midas_beit_large_512/colmap_0/<camera>/raw_depth_like.npy` 下提供 18 个文件；每个参考数组的 dtype 为 float32，形状为 `(2028, 2704)`。

完整参考布局为：

```text
/root/autodl-tmp/
├── coffee_martini/
│   ├── colmap_0/
│   ├── ...
│   ├── colmap_49/
│   ├── poses_bounds.npy
│   ├── dense_points.ply
│   └── midas_beit_large_512/
│       └── colmap_0/
│           ├── cam00/raw_depth_like.npy
│           ├── ...
│           └── cam20/raw_depth_like.npy
└── output/
    └── coffee_martini/
```

对于 N3D，运行器优先查找 `configs/n3d_ours/<scene>.json`。如果不存在，则使用 `configs/n3d_ours/default.json`。后者包含 50 帧 N3D 调度的最终模型，但关闭稠密初始化、背景补点和 MiDaS 过滤。Technicolor 使用专用的 `configs/tech_ours/<scene>.json`，没有通用回退配置。当持续时间、分辨率或其他训练参数与默认值不同时，应为新场景添加专用配置。

新场景主要涉及以下可选开关：

- 稠密初始化：将 `field_dense_initialization` 设为 `1`，保持 `field_dense_initialization_path` 为 `../dense_points.ply`。通常将 `field_dense_initialization_expected_points` 设为 `0`，除非必须验证特定产物的精确数量。
- 无 MiDaS 的背景补点：将 `field_bg_dense_add` 设为 `1`，保持 `field_bg_dense_beit_filter=0`。
- MiDaS 过滤的背景补点：将上述两个值都设为 `1`，一体化预处理脚本随后会生成所需深度类图。

## 训练、紧凑导出与测试

运行器默认使用 N3D 配置。一次普通调用会执行完整最终模型流程：训练完整模型、导出烘焙 static grid 的紧凑模型、评估请求的完整检查点，以及评估最终紧凑检查点。

```bash
python script/run_n3d_train_test.py --scene coffee_martini
```

Technicolor 需要显式选择对应配置。以下命令解析 `configs/tech_ours/Fabien.json`，使用 `technicolorvalid` 评估，读取 `/root/autodl-tmp/Fabien/colmap_0`，并写入 `/root/autodl-tmp/output/Fabien`：

```bash
python script/run_n3d_train_test.py --dataset tech --scene Fabien
```

Google Immersive 使用同一套最终模型训练、紧凑导出和测试流程。其配置选择标定鱼眼渲染器及 `immersivevalidss` 验证加载器。如果数据位于 `/root/autodl-tmp/Immersive`，例如：

```bash
python script/run_n3d_train_test.py \
  --dataset immersive \
  --scene 04_Truck \
  --datapath /root/autodl-tmp/Immersive \
  --savepath /root/autodl-tmp/output/Immersive
```

运行器会在启动实验前验证全部 50 个 COLMAP 切片、`models.json`、每个标定相机的采样网格，以及可选稠密初始化。完整模型评估和紧凑模型验证都会经过鱼眼映射；共享外观表示本身与其他数据集使用相同的 `coupled_detached` 最终模型。

位于默认数据根目录的其他场景可以直接启动：

```bash
python script/run_n3d_train_test.py --scene cook_spinach
```

使用通用默认配置时，只需要基础 COLMAP 序列，输入与输出解析为：

```text
/root/autodl-tmp/cook_spinach/colmap_0
/root/autodl-tmp/output/cook_spinach
```

通过 `--datapath` 和 `--savepath` 可以分别指定数据根目录和输出根目录；运行器会在两者后附加场景名称：

```bash
python script/run_n3d_train_test.py \
  --scene cook_spinach \
  --datapath /data/n3d \
  --savepath /data/stegf-output
```

上述命令读取 `/data/n3d/cook_spinach/colmap_0`，写入 `/data/stegf-output/cook_spinach`。旧参数名 `--data_root` 和 `--output_root` 仍作为别名接受。

标准训练调度为 30,000 步。解析后的最终迭代一定会保存并执行紧凑导出。N3D 使用 `colmapvalid` 评估，Technicolor 使用 `technicolorvalid`，Immersive 使用 `immersivevalidss`。默认 N3D 输出为：

```text
/root/autodl-tmp/output/coffee_martini
/root/autodl-tmp/output/coffee_martini/compact
```

不启动训练，仅验证路径和命令：

```bash
python script/run_n3d_train_test.py --scene coffee_martini --dry_run
```

运行一个不评估的短 CUDA 冒烟测试：

```bash
python script/run_n3d_train_test.py \
  --scene coffee_martini \
  --iterations 10 \
  --skip_test_stage \
  --skip_compact_stage
```

不重新训练，从已有完整检查点执行导出与评估：

```bash
python script/run_n3d_train_test.py \
  --scene coffee_martini \
  --skip_train_stage
```

不重新训练或导出，直接复用并评估已有紧凑模型：

```bash
python script/run_n3d_train_test.py \
  --scene coffee_martini \
  --skip_train_stage \
  --skip_compact_export
```

如果只需要旧的完整模型流程，传入 `--skip_compact_stage`。只有明确要替换相同最终迭代的已有紧凑产物时才使用 `--compact_overwrite`。`--skip_test_stage` 会跳过完整模型与紧凑模型评估，但仍执行紧凑导出，因为导出属于最终模型构建的一部分。

仍可直接调用底层完整检查点评估脚本：

```bash
python script/test_all_iterations.py \
  --iterations 30000 \
  --quiet --eval --skip_train \
  --valloader colmapvalid \
  --configpath configs/n3d_ours/coffee_martini.json \
  --model_path /root/autodl-tmp/output/coffee_martini \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0
```

### 手动导出紧凑推理模型

一体化运行器默认会执行该步骤。需要自定义输出位置或验证设置时，也可以直接调用导出器。最终 static appearance grid 在高斯规范位置查询，其训练相机平均贡献会被烘焙到逐高斯外观特征中，随后从保存的推理检查点删除网格：

```bash
python script/export_compact_model.py \
  --quiet --eval \
  --test_iteration 30000 \
  --valloader colmapvalid \
  --configpath configs/n3d_ours/coffee_martini.json \
  --model_path /root/autodl-tmp/output/coffee_martini \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0
```

源检查点不会被修改。导出器默认写入 `<model_path>/compact`，在内存中验证第 0、25、49 帧，保存 `compact_config.json`，并打印准确的评估命令。可通过 `--output_model_path` 指定其他目录；只有明确要替换同一迭代的已有紧凑检查点时才使用 `--overwrite`。紧凑产物仅用于推理，继续训练时必须使用原始检查点。

## 项目结构

- `train.py`：训练入口。
- `test.py`：渲染与评估入口。
- `script/run_n3d_train_test.py`：N3D、Technicolor、Immersive 一体化最终模型流程，使用 `--dataset n3d|tech|immersive` 选择数据集。
- `script/test_all_iterations.py`：检查点评估运行器。
- `script/preprocess_n3d_scene.py`：N3D 视频到模型就绪场景的一体化预处理。
- `script/preprocess_tech_scene.py`：Technicolor PNG 到模型就绪场景的一体化预处理。
- `script/preprocess_immersive_scene.py`：带标定的 Immersive 鱼眼视频预处理，生成针孔 COLMAP 切片并保留畸变域监督。
- `script/export_compact_model.py`：将 static grid 烘焙后的紧凑推理模型导出器。
- `script/setup_preprocess.sh`：独立的 COLMAP/Open3D/MiDaS 环境配置。
- `script/precompute_midas_beit_depth_like.py`：独立的可选 MiDaS 预处理。
- `configs/n3d_ours/default.json`：没有专用配置的 N3D 场景通用最终模型默认值。
- `configs/n3d_ours/coffee_martini.json`：最终参考实验配置。
- `configs/tech_ours`：Technicolor 最终模型场景配置。
- `configs/immersive_ours`：Google Immersive 最终模型场景配置。
- `thirdparty/gaussian_splatting`：高斯模型、渲染器和 CUDA 光栅化器源码。
- `helper_train.py`、`helper_model.py`：训练与模型工具。

## 致谢与引用

本项目基于 STGS 官方实现。一体化数据流水线还参考了 SpacetimeGaussians 的 N3D 稀疏预处理和 Google Immersive 标定鱼眼处理、E-D3DGS 的第 0 帧稠密重建与下采样流程，以及 MiDaS 的 `dpt_beit_large_512` 模型。相关上游许可证条款继续适用。

```bibtex
@inproceedings{li2024stgs,
  title={Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
  author={Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
  booktitle={CVPR},
  year={2024}
}
```

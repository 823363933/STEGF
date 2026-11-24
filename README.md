# Shared Appearance Scaffold for Dynamic 3D Gaussian Splatting

English | [简体中文](README_zh-CN.md)

## Overview

Dynamic 3D Gaussian splatting must represent both stable scene appearance and
time-varying detail. Per-Gaussian appearance alone does not explicitly share
repeated evidence through a spatial structure, while independently optimized
motion and temporal opacity can make motion complexity inconsistent with a
primitive's visible lifetime.

This project adds a multiresolution feature grid as a **shared appearance
scaffold**. Each Gaussian queries the grid at its canonical position and
combines the shared feature with its own local appearance feature. A detached
motion–lifetime coupling scales quadratic and cubic motion by temporal support:
short-lived Gaussians approach linear motion, while long-lived Gaussians retain
higher-order trajectories. After training, the scaffold contribution can be
baked into the Gaussian features, so the compact inference model no longer
needs the grid or its query computation.

## Reconstruction examples

<table>
  <tr>
    <td width="50%"><img src="assets/cut_roasted_beef_interactive_4d.gif" alt="cut_roasted_beef" width="480" height="380"></td>
    <td width="50%"><img src="assets/02_flames_interactive_4d.gif" alt="02_Flames" width="480" height="380"></td>
  </tr>
</table>

## Installation

Run commands from the repository root. Training requires Conda, an NVIDIA GPU,
and a CUDA compiler compatible with PyTorch 2.0.0 and CUDA 11.8.

```bash
bash script/setup.sh
conda activate STEGF
```

The setup script installs the training environment and builds the local
rasterization, simple-KNN, and MMCV KNN extensions.

Dataset preprocessing uses a separate environment:

```bash
# RTX 50 series: CUDA 12.8 / Python 3.10 profile
bash script/setup_preprocess.sh

# RTX 40 series: CUDA 11.8 / Python 3.8 profile
bash script/setup_preprocess_server.sh

conda activate STEGF-preprocess
```

Both profiles install COLMAP 4.1.1, Open3D, and image-processing dependencies.
MiDaS and its weights are installed by default; append `--without-midas` when
the scene configuration does not use MiDaS priors. On a headless server,
install `xvfb` and `xauth`; the preprocessors automatically use `xvfb-run`
when COLMAP needs a display.

## Data preparation

This work has been evaluated on Neural 3D, Technicolor, and Google Immersive.
All three datasets use the same processed scene layout:

```text
<datapath>/
└── <scene>/
    ├── colmap_0/
    ├── colmap_1/
    ├── ...
    ├── colmap_<duration-1>/
    ├── dense_points.ply                 # optional downsampled dense initialization
    └── midas_beit_large_512/            # optional depth priors
```

Each `colmap_<time>` contains synchronized images and a COLMAP model under
`sparse/0`. The full time sequence is required. Optional data is needed only
when the corresponding `field_*` options are enabled in the scene configuration.

| Dataset | Required raw data and calibration | Preprocessor | Configurations |
| --- | --- | --- | --- |
| N3D / DyNeRF | synchronized `cam*.mp4`, `poses_bounds.npy` | `preprocess_n3d_scene.py` | `configs/n3d_ours` |
| Technicolor | official PNG sequences, `cameras_parameters.txt` | `preprocess_tech_scene.py` | `configs/tech_ours` |
| Google Immersive | calibrated camera MP4s, `models.json` | `preprocess_immersive_scene.py` | `configs/immersive_ours` |

### Neural 3D

Download the dataset from [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video.git),
then run preprocessing:

```bash
conda activate STEGF-preprocess
python script/preprocess_n3d_scene.py \
  --scene <scene name> \
  --inputpath <source>/DyNeRF \
  --datapath <pre path>/DyNeRF-pre
```

### Technicolor

Contact the authors of *Dataset and Pipeline for Multi-View Light-Field Video*
to obtain the Technicolor dataset.

```bash
conda activate STEGF-preprocess
python script/preprocess_tech_scene.py \
  --scene <scene name> \
  --inputpath <source>/Tech \
  --datapath <source>/Tech-pre
```

### Google Immersive

Download the dataset from [DeepView Video Dataset](https://github.com/augmentedperception/deepview_video_dataset).

```bash
conda activate STEGF-preprocess
python script/preprocess_immersive_scene.py \
  --scene <scene name> \
  --inputpath <source>/Immersive \
  --datapath <source>/Immersive-pre
```

## Training and evaluation

`script/run_n3d_train_test.py` is the common runner for all three datasets.
A standard run performs four stages:

1. Train the full model.
2. Evaluate the requested full checkpoints.
3. Bake the shared scaffold into a compact inference model.
4. Evaluate the final compact checkpoint.

### N3D

```bash
python script/run_n3d_train_test.py \
  --scene <scene name> \
  --datapath <pre path>/DyNeRF-pre \
  --savepath output/DyNeRF
```

### Technicolor

```bash
python script/run_n3d_train_test.py \
  --dataset tech \
  --scene <scene name> \
  --datapath <pre path>/Tech-pre \
  --savepath output/Tech
```

### Google Immersive

```bash
python script/run_n3d_train_test.py \
  --dataset immersive \
  --scene <scene name> \
  --datapath <pre path>/Immersive-pre \
  --savepath output/Immersive
```

By default, N3D and Technicolor scenes resolve under `/root/autodl-tmp`.
Other data roots can always be supplied explicitly.

Evaluation saves one shared GT set at `<savepath>/<scene>/gt/test/`.
All checkpoints, compact evaluations and photometric fits reuse it; rendered
and fitted images remain in their respective checkpoint directories.

### Paper checkpoints

[Paper run settings](configs/paper_runs.json) records the scene configuration,
original training budget, and selected checkpoint for each protocol:
`n3d300`, `n3d50`, `tech` (50 frames), and `immersive` (50 frames).
Pass the corresponding scene configuration to the preprocessor with
`--configpath` (in particular for the 300-frame N3D data).

```bash
python script/reproduce_paper.py --profile n3d300 --scene cook_spinach \
  --datapath /path/to/N3D-300 --savepath output/paper
```

Use `--dry_run` to print the commands without running them. After training,
the runner exports and evaluates the **selected** compact checkpoint under
`<savepath>/<profile>/<scene>/compact/`, without fitting to held-out images.
Data and trained weights are not included. Exact numerical reproduction also
depends on the prepared inputs; the historical 50-frame `coffee_martini`
result used different COLMAP and frame-0 initialization inputs from those
produced by the current preprocessing recipe.

## Compact inference model

The standard runner automatically exports the compact model. The exporter
queries the trained scaffold at canonical Gaussian positions, bakes the mean
contribution over training cameras into per-Gaussian appearance features,
removes training-only grid and routing parameters, and writes:

```text
<savepath>/<scene>/compact/
```

## Acknowledgements

Our data pipeline follows the STGS preprocessing and Immersive fisheye
handling, and the frame-0 dense reconstruction and downsampling procedure
of E-D3DGS. Depth priors use the `dpt_beit_large_512` model from MiDaS.
We thank the maintainers of these open-source projects:
[STGS](https://github.com/oppo-us-research/SpacetimeGaussians),
[E-D3DGS](https://github.com/JeongminB/E-D3DGS),
[MiDaS](https://github.com/isl-org/MIDAS).

## License

Original code contributions by **STEGF contributors** are licensed under MIT;
see [LICENSE](LICENSE). Upstream code and modifications to it retain their
applicable licenses, including the Gaussian Splatting non-commercial research
and evaluation restrictions. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution and component-specific terms. Datasets, pretrained weights,
and dataset-derived demonstration media retain their respective terms.

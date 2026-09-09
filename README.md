# STEGF: Spatio-Temporal Euler-Gaussian Field

This repository contains the final STEGF research model for the Neural 3D
`coffee_martini` scene. The selected version is
`S2.0.4-couptest-poly` with `coupled_detached` motion-opacity coupling.

## Final model

- Representation: dynamic 3D Gaussian primitives with a static multilevel
  Euler appearance field.
- Initialization: all-time COLMAP sparse points plus a frame-0 dense point
  cloud.
- Existence: transient Gaussian time kernels initialized to cover nearly the
  full sequence.
- Motion: per-Gaussian linear, quadratic, and cubic coefficients. The
  quadratic and cubic displacement is scaled by the detached normalized time
  width, so motion loss does not directly change the lifetime width.
- Appearance: static Euler-grid residual and trainable luma/white-balance
  exposure correction.
- Disabled final branches: dynamic Euler grid, staged routing, EMS guided
  sampling, temporal refinement, Carrier motion, spatiotemporal bubbles, and
  piecewise-velocity motion.

The main scene configuration is
`configs/n3d_ours/coffee_martini.json`.

## Environment

Run all commands from the repository root. A Conda installation, an NVIDIA
driver, and a working CUDA compiler are required because the rasterizer,
simple-KNN, and MMCV KNN operators are built locally.

```bash
bash script/setup.sh
conda activate STEGF
```

The setup script creates or updates the Python 3.8 `STEGF` Conda environment,
installs PyTorch 2.0.0, torchvision 0.15.0, and torchaudio 2.0.0 from the CUDA
11.8 wheel index, builds the three local extensions, and verifies that they can
be imported. A CUDA compiler compatible with this PyTorch build must be
available as `nvcc` while the extensions are built.

## Required data

The standard runner assumes the following external data layout. These assets
are intentionally not stored in Git.

```text
/root/autodl-tmp/
├── coffee_martini/
│   ├── colmap_0/
│   ├── colmap_1/
│   ├── ...
│   ├── colmap_49/
│   └── midas_beit_large_512/
│       └── colmap_<time>/camXX/raw_depth_like.npy
└── coffee_martini_ed3dgs50_ready/
    └── points3D_downsample.ply
```

`points3D_downsample.ply` is the frame-0 dense initialization and must contain
exactly 97,877 points for the final `coffee_martini` configuration.

The MiDaS-BEiT depth-like files are used by the background densification event.
If they are not already available, generate them with the retained preprocessing
script and an external MiDaS checkout:

```bash
python script/precompute_midas_beit_depth_like.py \
  --scene coffee_martini \
  --dataset_root /root/autodl-tmp \
  --output_dir /root/autodl-tmp/coffee_martini/midas_beit_large_512 \
  --time_indices all \
  --midas_repo /root/OriginalRepository/MiDaS
```

## Train and test

The complete standard workflow is:

```bash
python script/run_n3d_train_test.py --scene coffee_martini
```

It uses the standard 30,000-iteration training schedule, saves the final
checkpoint, and evaluates it with the `colmapvalid` loader. The default output
directory is:

```text
/root/autodl-tmp/output/coffee_martini-coupled-detached
```

Validate paths and commands without launching training:

```bash
python script/run_n3d_train_test.py --scene coffee_martini --dry_run
```

Run a short CUDA smoke test without evaluation:

```bash
python script/run_n3d_train_test.py \
  --scene coffee_martini \
  --iterations 10 \
  --skip_test_stage
```

Evaluate an existing final checkpoint:

```bash
python script/test_all_iterations.py \
  --iterations 30000 \
  --quiet --eval --skip_train \
  --valloader colmapvalid \
  --configpath configs/n3d_ours/coffee_martini.json \
  --model_path /root/autodl-tmp/output/coffee_martini-coupled-detached \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0
```

## Project layout

- `train.py`: training entry point.
- `test.py`: rendering and evaluation entry point.
- `script/run_n3d_train_test.py`: standard final-model workflow.
- `script/test_all_iterations.py`: checkpoint evaluation runner.
- `script/precompute_midas_beit_depth_like.py`: required auxiliary-data
  preprocessing.
- `configs/n3d_ours/coffee_martini.json`: final experiment configuration.
- `thirdparty/gaussian_splatting`: Gaussian model, renderer, and CUDA
  rasterizer sources.
- `helper_train.py`, `helper_model.py`: training and model utilities.

## Attribution

This project is based on the official STGS implementation:

```bibtex
@inproceedings{li2024stgs,
  title={Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
  author={Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
  booktitle={CVPR},
  year={2024}
}
```

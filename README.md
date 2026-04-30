# STEGF: Spatio-Temporal Euler-Gaussian Field

STEGF is a research codebase derived from STGS. This repository is trimmed to the V2.6.4 project starting point: STGS dynamic Gaussians are kept as the main representation, and a static multi-level Euler grid adds a 6D appearance residual before rasterization.

## V2.6.4 Scope

- Supported model path: `ours_full`
- Supported loader path: Neural 3D style `colmap` / `colmapvalid`
- Supported renderer path: `train_ours_full` / `test_ours_full`
- Enabled: static Euler grid 6D appearance residual
- Disabled: dynamic grid, EMS main guided sampling, omega/rotation split freeze, temporal refine, staged routing, static geometry/opacity residuals

## Install

The project expects a CUDA/PyTorch environment compatible with the original STGS setup. The runtime CUDA extensions used by this trimmed version are:

```bash
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
pip install thirdparty/gaussian_splatting/submodules/simple-knn
pip install -e thirdparty/mmcv -v
```

Additional Python packages used by training/testing include `torch`, `torchvision`, `opencv-python`, `tqdm`, `numpy`, `scipy`, `scikit-image`, `natsort`, `kornia`, `plyfile`, and `Pillow`.

## Train

```bash
# coffee_martini
python train.py --quiet --eval \
  --configpath configs/n3d_ours/coffee_martini.json \
  --model_path /root/autodl-tmp/output/coffee_martini \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0 \
  --save_iterations 30000

# cook_spinach
python train.py --quiet --eval \
  --configpath configs/n3d_ours/cook_spinach.json \
  --model_path /root/autodl-tmp/output/cook_spinach \
  --source_path /root/autodl-tmp/cook_spinach/colmap_0 \
  --save_iterations 30000
```

## Test

```bash
# coffee_martini
python script/test_all_iterations.py --quiet --eval --skip_train \
  --valloader colmapvalid \
  --configpath configs/n3d_ours/coffee_martini.json \
  --model_path /root/autodl-tmp/output/coffee_martini \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0

# cook_spinach
python script/test_all_iterations.py --quiet --eval --skip_train \
  --valloader colmapvalid \
  --configpath configs/n3d_ours/cook_spinach.json \
  --model_path /root/autodl-tmp/output/cook_spinach \
  --source_path /root/autodl-tmp/cook_spinach/colmap_0
```

## Project Layout

- `train.py`: training entry point
- `test.py`: evaluation/rendering entry point
- `script/test_all_iterations.py`: tests all saved point-cloud iterations under one model path
- `configs/n3d_ours`: STEGF V2.6.4 scene configs
- `thirdparty/gaussian_splatting`: STGS/3DGS runtime code and CUDA rasterizer sources
- `helper_train.py`, `helper_model.py`: training/model utility code

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

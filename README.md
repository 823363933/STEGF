# STEGF: Spatio-Temporal Euler-Gaussian Field

This repository contains the final STEGF research model. The selected version
is `S2.0.4-couptest-poly` with `coupled_detached` motion-opacity coupling. The
Neural 3D Video `coffee_martini` scene is the reference configuration and
reproduction case, not a hard-coded model restriction.

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

Video preprocessing uses a separate environment because COLMAP, Open3D, and
MiDaS are not training-time dependencies:

```bash
bash script/setup_preprocess.sh
conda activate STEGF-preprocess
```

The default preprocessing profile targets a local Python 3.10 environment and
installs PyTorch 2.7.1 from the CUDA 12.8 wheel index. For a CUDA 11.8 server
whose training stack uses Python 3.8 and PyTorch 2.0.0, use the separate server
profile instead:

```bash
bash script/setup_preprocess_server.sh
conda activate STEGF-preprocess
```

The server profile installs Python 3.8, PyTorch 2.0.0, torchvision 0.15.0,
and torchaudio 2.0.0 from the CUDA 11.8 wheel index. Both profiles install
CUDA-enabled COLMAP 4.1.1 and the required image, point-cloud, and
depth-inference packages. They also place the pinned MiDaS
source and `dpt_beit_large_512.pt` weights in the Git-ignored
`thirdparty/MiDaS/` directory. Once preprocessing is complete, neither Open3D
nor MiDaS is needed during model training. For a configuration that will not
generate MiDaS priors, append `--without-midas` to either setup command to skip
its PyTorch dependencies, source checkout, and model download.

On a headless Ubuntu server, install a virtual display once before running the
scene preprocessor:

```bash
apt-get update
apt-get install -y xvfb xauth
```

When a COLMAP stage is still required and `DISPLAY` is unset,
`preprocess_n3d_scene.py` automatically restarts itself under `xvfb-run -a`.
It does not use Xvfb when COLMAP outputs are already complete and only a later
stage such as MiDaS remains.

## Dataset convention

The final N3D runner uses a scene-independent layout. Let `<location>` be the
data root passed with `--datapath`, and let `<scene>` be the scene name passed
with `--scene`:

```text
<location>/
├── <scene>/
│   ├── colmap_0/
│   ├── colmap_1/
│   ├── ...
│   ├── colmap_<duration-1>/
│   ├── poses_bounds.npy
│   ├── dense_points.ply                 # optional
│   └── midas_beit_large_512/            # optional
└── output/
    └── <scene>/
```

The default `<location>` is `/root/autodl-tmp`. The standard command passes
`<location>/<scene>/colmap_0` as `--source_path`; the loader discovers the
remaining time slices as sibling `colmap_*` directories. Unless `--savepath`
is provided explicitly, outputs are written to
`<location>/output/<scene>`.

The indispensable scene data is `poses_bounds.npy` and the complete sequence
of `colmap_<time>` directories. `dense_points.ply` is required only when
`field_dense_initialization=1`. `midas_beit_large_512/` is required only when
background point insertion and its MiDaS filter are both enabled. The generic
`default.json` disables both optional mechanisms; the dedicated
`coffee_martini.json` enables both for exact reproduction of the reference
model.

### One-command preprocessing from videos

The integrated preprocessor accepts one calibrated N3D/DyNeRF-style scene:

```text
<raw-location>/<scene>/
├── cam00.mp4
├── cam01.mp4
├── ...
└── poses_bounds.npy
```

The videos must be synchronized, and the rows of `poses_bounds.npy` must
correspond to `cam*.mp4` after filename sorting. The script consumes existing
LLFF/N3D camera calibration; it does not infer camera poses for uncalibrated
videos.

Prepare every artifact enabled by the resolved scene configuration with one
command:

```bash
conda activate STEGF-preprocess
python script/preprocess_n3d_scene.py \
  --scene <scene> \
  --inputpath <raw-location> \
  --datapath <location>
```

For example, a local N3D download under `dataset/DyNeRF` can be prepared with:

```bash
python script/preprocess_n3d_scene.py \
  --scene coffee_martini \
  --inputpath dataset/DyNeRF \
  --datapath /root/autodl-tmp
```

The pipeline performs these stages in order:

1. It extracts the configured sequence from each camera video and performs the
   SpacetimeGaussians-style fixed-pose COLMAP reconstruction independently at
   every time index.
2. If dense initialization is enabled, it runs COLMAP PatchMatch and fusion on
   the multi-view images at time 0, then applies the E-D3DGS iterative voxel
   downsampling rule until no more than 100,000 points remain.
3. If MiDaS-filtered background insertion is enabled, it generates
   `dpt_beit_large_512` depth-like arrays for the time indices in
   `field_bg_dense_add_time_indices`.

`--dense auto` and `--midas auto` are the defaults and follow the resolved
configuration (`<scene>.json`, then `default.json`). `--dense on|off` and
`--midas on|off` override artifact generation without editing the model
configuration. Other useful controls include `--duration`,
`--source_start_frame`, `--downscale`, `--gpu_index`, and `--dry_run`.

Completed `colmap_<time>` results are skipped. An interrupted hidden build
workspace is replaced only when `--restart_incomplete` is supplied; an
incomplete published `colmap_<time>` is never deleted automatically.

### Per-time RGB images and sparse point clouds

For a sequence with `duration=N`, the scene must contain `colmap_0` through
`colmap_<N-1>`. Each time slice has this minimum structure:

```text
<location>/<scene>/colmap_<time>/
├── images/
│   ├── <camera-0>.png
│   ├── <camera-1>.png
│   └── ...
└── sparse/0/
    └── points3D.bin
```

Every time slice must contain the same camera image names. In addition,
`colmap_0/sparse/0/` must contain `cameras.bin` and `images.bin`, whose image
names must correspond to the PNG files. The scene-level `poses_bounds.npy` is
also required.

The loader concatenates `points3D.bin` from all time slices into
`colmap_0/sparse/0/points3D_total<N>.ply`. This file is an optional generated
cache; if it is absent, all source `points3D.bin` files must exist and the
`colmap_0/sparse/0` directory must be writable.

With `--eval`, the first camera after natural name sorting is used as the test
view for every time step; all remaining cameras are training views. The
original COLMAP `distorted`, `manual`, `stereo`, `tmp`, and `input.db` entries
are not read by the final path.

### Dense frame-0 initialization

`<location>/<scene>/dense_points.ply` is the optional dense point cloud for
time index 0. It is fused with the all-time sparse points during model
initialization and is a hard requirement only when
`field_dense_initialization=1`.

The PLY must contain `x`, `y`, `z` and either `red`, `green`, `blue` or `r`,
`g`, `b` properties. Its expected vertex count is scene-specific: set
`field_dense_initialization_expected_points` in the scene configuration to the
known count, or to `0` to disable only the count check. The standard relative
configuration path is `../dense_points.ply`, resolved from `colmap_0`. The
97,877-point value belongs only to the historical `coffee_martini` artifact;
other scenes are not required to produce that exact count.

### MiDaS-BEiT depth-like arrays

When both `field_bg_dense_add=1` and `field_bg_dense_beit_filter=1`, each
background-densification time and camera used by the scene configuration
needs:

```text
<location>/<scene>/midas_beit_large_512/
└── colmap_<time>/
    ├── <camera-0>/raw_depth_like.npy
    ├── <camera-1>/raw_depth_like.npy
    └── ...
```

The arrays are float32 depth-like maps. Training resizes them to the current
render resolution when necessary. The preprocessing diagnostics
`band_p10_p030.png`, `stats.txt`, and `manifest_multitime.json` are not read by
training.

The integrated scene preprocessor generates these files automatically when the
configuration enables them. They can also be generated independently:

```bash
python script/precompute_midas_beit_depth_like.py \
  --scene <scene> \
  --datapath <location> \
  --time_indices <comma-separated-time-indices>
```

The standalone script uses `thirdparty/MiDaS`, writes to
`<location>/<scene>/midas_beit_large_512` by default, and accepts
`--midas_repo` for another checkout. MiDaS and its weights are not required
during training after the arrays have been generated. A missing directory
currently makes training disable the filter with a warning, but such a run is
not equivalent to the enabling configuration and represents incomplete input.

### Reference `coffee_martini` data

The committed reference configuration uses `duration=50` and therefore
expects `colmap_0` through `colmap_49`. Each time slice contains these 18
camera images:

```text
cam00.png  cam01.png  cam02.png  cam04.png  cam05.png  cam06.png
cam07.png  cam08.png  cam09.png  cam10.png  cam11.png  cam12.png
cam13.png  cam14.png  cam16.png  cam18.png  cam19.png  cam20.png
```

The reference configuration consumes 900 images at 2704 by 2028 pixels. With
`--eval`, `cam00` supplies 50 test frames and the other 17 cameras supply 850
training frames. The historical final-experiment inputs contained 97,877
dense vertices and 344,575 all-time sparse points. Fresh CUDA COLMAP runs are
not expected to reproduce those point counts byte-for-byte: feature matching,
triangulation, dense fusion, and voxel downsampling can produce small run-to-run
differences from the same images. The committed configuration therefore sets
the dense exact-count check to `0`; the integrated preprocessor instead
enforces the configured 100,000-point upper bound and validates the required
point attributes.

The final configuration runs its BEiT-filtered background densification event
only at time index 0. Exact reproduction therefore needs 18 files under
`midas_beit_large_512/colmap_0/<camera>/raw_depth_like.npy`; each reference
array has dtype float32 and shape `(2028, 2704)`.

The complete reference layout is:

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

For configuration lookup, the runner first tries
`configs/n3d_ours/<scene>.json`. If it does not exist, it uses
`configs/n3d_ours/default.json`, which contains the final model with a
50-frame N3D schedule but disables dense initialization, background point
insertion, and the MiDaS filter. Copy it to `<scene>.json` and enable only the
optional mechanisms for which the scene has been preprocessed. A
scene-specific configuration should also be added when duration, resolution,
or other scene-dependent training settings differ from the defaults.

For a new scene, the relevant optional switches are:

- Dense initialization: set `field_dense_initialization` to `1`, keep
  `field_dense_initialization_path` as `../dense_points.ply`, and normally set
  `field_dense_initialization_expected_points` to `0` unless an exact artifact
  count must be enforced.
- Background insertion without MiDaS: set `field_bg_dense_add` to `1` and
  leave `field_bg_dense_beit_filter` at `0`.
- MiDaS-filtered background insertion: set both values to `1`; the integrated
  preprocessor then generates the required depth-like maps automatically.

## Train and test

The complete standard workflow is:

```bash
python script/run_n3d_train_test.py --scene coffee_martini
```

Another scene stored under the default data root can be started directly:

```bash
python script/run_n3d_train_test.py --scene cook_spinach
```

With the generic default configuration this requires only the base COLMAP
sequence and resolves the input and output paths as:

```text
/root/autodl-tmp/cook_spinach/colmap_0
/root/autodl-tmp/output/cook_spinach
```

Use independent roots for datasets and results with `--datapath` and
`--savepath`. The runner appends the scene name to both roots:

```bash
python script/run_n3d_train_test.py \
  --scene cook_spinach \
  --datapath /data/n3d \
  --savepath /data/stegf-output
```

This reads `/data/n3d/cook_spinach/colmap_0` and writes
`/data/stegf-output/cook_spinach`. The older names `--data_root` and
`--output_root` remain accepted as aliases.

It uses the standard 30,000-iteration training schedule, saves the final
checkpoint, and evaluates it with the `colmapvalid` loader. The default output
directory is:

```text
/root/autodl-tmp/output/coffee_martini
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
  --model_path /root/autodl-tmp/output/coffee_martini \
  --source_path /root/autodl-tmp/coffee_martini/colmap_0
```

## Project layout

- `train.py`: training entry point.
- `test.py`: rendering and evaluation entry point.
- `script/run_n3d_train_test.py`: standard final-model workflow.
- `script/test_all_iterations.py`: checkpoint evaluation runner.
- `script/preprocess_n3d_scene.py`: integrated N3D video-to-STEGF scene
  preprocessing.
- `script/setup_preprocess.sh`: separate COLMAP/Open3D/MiDaS environment
  setup.
- `script/precompute_midas_beit_depth_like.py`: standalone optional MiDaS
  preprocessing.
- `configs/n3d_ours/default.json`: generic final-model defaults for N3D
  scenes without a dedicated configuration.
- `configs/n3d_ours/coffee_martini.json`: final experiment configuration.
- `thirdparty/gaussian_splatting`: Gaussian model, renderer, and CUDA
  rasterizer sources.
- `helper_train.py`, `helper_model.py`: training and model utilities.

## Attribution

This project is based on the official STGS implementation. The integrated
data pipeline also follows the N3D sparse preprocessing of
SpacetimeGaussians, the frame-0 dense reconstruction/downsampling procedure of
E-D3DGS, and the `dpt_beit_large_512` model from MiDaS; their upstream license
terms continue to apply.

```bibtex
@inproceedings{li2024stgs,
  title={Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
  author={Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
  booktitle={CVPR},
  year={2024}
}
```

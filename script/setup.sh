#!/bin/bash
set -e

conda create -n stegf python=3.7.13 -y
conda activate stegf

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y

pip install opencv-python tqdm natsort scipy kornia plyfile Pillow scikit-image

pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
pip install thirdparty/gaussian_splatting/submodules/simple-knn
pip install -e thirdparty/mmcv -v

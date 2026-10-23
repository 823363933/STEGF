# Third-party notices

Original STEGF source-code contributions are Copyright (c) 2026 STEGF
contributors and are offered under the MIT terms in [LICENSE](LICENSE), except
where another license applies. This notice records upstream code, adaptations,
and external tools used by the project; it does not replace their licenses.
File-level notices and the full upstream license texts remain applicable.

## SpacetimeGaussians (STGS)

- Source: [oppo-us-research/SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians).
- Copyright (c) 2023 OPPO.
- Scope: the training and evaluation foundation, helpers, and preprocessing
  code inherited or adapted from STGS.
- License: MIT for the portions so licensed by STGS, together with its
  additional Gaussian Splatting use limitation. Both the OPPO copyright and
  the original additional-use notice are retained in [LICENSE](LICENSE).
- Upstream text: [STGS LICENSE](https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/LICENSE).

The MIT grant in STGS does not relicense its Gaussian Splatting-derived code.
The same distinction applies to this repository.

## Gaussian Splatting, rasterization, and simple-KNN

- Sources: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting),
  [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization),
  and [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn), inherited through STGS.
- Copyright notices identify Inria and the GRAPHDECO research group; the
  Gaussian-Splatting License identifies Inria and MPII as the licensors.
- Scope: inherited Gaussian models, camera and scene utilities, renderer code,
  and their derivatives under `thirdparty/gaussian_splatting/`, including
  `submodules/gaussian_rasterization_ch9/` and `submodules/simple-knn/`.
- Retained full licenses:
  [Gaussian-Splatting License](thirdparty/gaussian_splatting/LICENSE.md) and
  [rasterization license](thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9/LICENSE.md).

The simple-KNN sources retain their Inria headers referring to these research
and evaluation terms. They are not covered by a blanket MIT grant. Separately
licensed files and components inside these directories retain their own terms,
including GLM, LPIPS, and the SSIM implementation identified below.

The Gaussian-Splatting License limits use to non-commercial research and
evaluation and requires preservation of its license and attribution when
redistributing the covered work. Its terms continue to apply to modifications
of that work. Commercial use requires prior and explicit consent from the
applicable upstream licensors. See the full retained license for the governing
terms and contact details.

## OpenMMLab MMCV

- Source: [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv).
- Copyright (c) OpenMMLab. All rights reserved.
- Location: `thirdparty/mmcv/`; the training setup builds its KNN operation.
- Principal license: [Apache License 2.0](thirdparty/mmcv/LICENSE).
- Component exceptions: [LICENSES.md](thirdparty/mmcv/LICENSES.md).

Some bundled operations have separate NVIDIA license terms. The Apache 2.0
label does not override those operation-specific notices, whether or not the
standard STEGF training path uses those operations.

## OpenGL Mathematics (GLM)

- Source: [g-truc/glm](https://github.com/g-truc/glm).
- Copyright (c) 2005 - G-Truc Creation.
- Location: `thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9/third_party/glm/`.
- License: the bundled version offers the Happy Bunny License or the MIT
  License. The complete text of both alternatives is retained in
  [copying.txt](thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9/third_party/glm/copying.txt).

## LPIPS implementation and downloaded weights

The implementation in `thirdparty/gaussian_splatting/lpipsPyTorch/` is inherited
through STGS and corresponds to
[S-aiueo32/lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch).
Its [upstream license](https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE)
is reproduced here:

```text
BSD 2-Clause License

Copyright (c) 2020, Sou Uchida
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The metric downloads linear weights from
[richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
and feature-network weights through torchvision. The upstream
[PerceptualSimilarity license](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE)
is reproduced below. Downloaded feature-network weights retain their own
applicable terms; the STEGF MIT grant does not relicense them.

```text
Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## SSIM implementation

The SSIM implementation in `thirdparty/gaussian_splatting/utils/loss_utils.py`
is based on [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim), inherited
through Gaussian Splatting. Gaussian Splatting attributes this code to
Copyright Evan Su, 2017, under MIT; see its
[upstream license, section 6](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).
The MIT permission and disclaimer text is included in [LICENSE](LICENSE).
This attribution does not change the licensing of other code in that file.

## Preprocessing references and external software

- **E-D3DGS:** [JeongminB/E-D3DGS](https://github.com/JeongminB/E-D3DGS)
  provides the reference workflow for frame-0 dense reconstruction and voxel
  downsampling used in `script/preprocess_n3d_scene.py` and reused by the other
  preprocessors. Its repository-level
  [license](https://github.com/JeongminB/E-D3DGS/blob/main/LICENSE.md) is the
  Gaussian-Splatting License. Existing file-level OPPO/MIT notices in inherited
  preprocessing code do not make the E-D3DGS repository as a whole MIT-licensed;
  any adapted code retains its applicable upstream terms.
- **MiDaS:** [isl-org/MiDaS](https://github.com/isl-org/MiDaS) is downloaded
  separately by the optional preprocessing setup. Its source is Copyright (c)
  2019 Intel ISL (Intel Intelligent Systems Lab), under its
  [MIT license](https://github.com/isl-org/MiDaS/blob/master/LICENSE).
  Keep the checkout's license when redistributing it. The selected
  `dpt_beit_large_512` checkpoint and its dependencies are not relicensed by
  the STEGF source-code license.
- **COLMAP:** [colmap/colmap](https://github.com/colmap/colmap) is installed
  separately for reconstruction. Database-layout/helper adaptations in the
  preprocessors follow COLMAP's Python database helper. COLMAP's own code uses
  its [BSD license](https://colmap.github.io/license.html);
  its dependencies retain their separate licenses. Existing COLMAP-related
  loaders inherited through Gaussian Splatting also retain their file headers.
- **Other installed tools:** PyTorch, torchvision, Open3D, OpenCV, MiDaS
  dependencies, FFmpeg, and the optional Viser viewer are installed separately
  and retain their own licenses. A binary distribution must retain the
  applicable notices for the actual versions and components it includes.

### COLMAP database-helper attribution

For COLMAP-derived database-helper portions of the preprocessors, the upstream
BSD notice from the COLMAP version inherited through STGS is reproduced below.
Separately installed COLMAP versions retain their own distributed notices.
Source: [COLMAP 3.8 COPYING.txt](https://github.com/colmap/colmap/blob/3.8/COPYING.txt).

```text
Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
  its contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Datasets, checkpoints, and demonstration media

N3D/DyNeRF, Technicolor, and Google Immersive datasets, pretrained weights, and
dataset-derived demonstration images or videos are not granted an MIT license
by this repository. They remain subject to their respective rights and terms.
Use the datasets' official distribution channels and terms. Publishing or
downloading a checkpoint or a GIF does not by itself change those terms.

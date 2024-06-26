
<div align='center'>

<h2>Point-SAM: Promptable 3D Segmentation Model for Point Clouds</h2>


[Yuchen Zhou](https://github.com/zyc00) <sup>\* 1</sup>, [Jiayuan Gu](https://cseweb.ucsd.edu/~jigu/) <sup>\* 1</sup>, [Tung Yen Chiang](https://github.com/tungyen)<sup>1</sup>, [Fanbo Xiang](https://www.fbxiang.com)<sup>1</sup>, [Hao Su](https://cseweb.ucsd.edu/~haosu/)<sup>1,2</sup>

UC San Diego, Hillbot

<sup>* </sup>Equal Contribution

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2406.17741-red)](https://arxiv.org/abs/2406.17741)
[![Project Page](https://img.shields.io/badge/Project-page-yellow
)](https://point-sam.github.io)
[![Demo](https://img.shields.io/badge/Demo-huggingface-orange
)](https://huggingface.co/spaces/yuchen0187/Point-SAM)

</div>


<p>
  <img src="./assets/teaser.jpeg" width="100%"/>
</p>

### Installation
The code requires `python>=3.8`, `timm>=0.9.0`, `pytorch>=2.1.0`, `torchvision>=0.16.0`. Please follow the offcial guide to install Pytorch, timm and TorchVision dependencies. We also recommend compiling third partiy modules with `g++=9.3.0`.

Install third party modules.
```
# Install g++=9.3.0 in conda environment by
conda install gxx_linux-64=9.3.0
# Install torkit3d
git submodule update --init third_party/torkit3d && 
FORCE_CUDA=1 pip install third_party/torkit3d &&
# Install apex
git submodule update --init third_party/apex &&
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" third_party/apex
```


Install Point-SAM from GitHub
```
pip install git+https://github.com/zyc00/Point-SAM.git
```
or clone the repository and install with
```
git clone https://github.com/zyc00/Point-SAM.git &&
cd Point-SAM && pip install . && cd ..
```

### Getting Start
We provide pretrained [checkpoint](https://huggingface.co/yuchen0187/Point-SAM/tree/main) for Point-SAM with ViT-L on HuggingFace. After downloading the checkpoint, you can inference with the following codes. For different numbers of points, we allow custom `group_number` and `group_size`. The default setting is `group_number=512` and `group_size=64`, and we suggest setting `group_number=2048` and `group_size=256` while the number of points > 100k.
```
from point_sam import build_point_sam
model = build_point_sam(ckpt_path, group_number, group_size)
model.set_pointcloud(coords, colors)
model.predict_masks(prompt_coords, prompt_labels)
```
### Mesh Segmentation Demo

Our model support mesh segmentation by sampling points from the mesh and propagate segmentation labels back to the mesh. We deployed a huggingface demo at 🤗[demo](https://huggingface.co/spaces/yuchen0187/Point-SAM)🤗. We also provide local mesh segmentation demo at https://github.com/zyc00/point-sam-demo. Some meshes in glb format are provided in [examples](https://github.com/zyc00/point-sam-demo/tree/main/examples).

<p>
  <img src="./assets/indoor-ezgif.com-video-to-gif-converter.gif" width="49%"/>
  <img src="./assets/outdoor-ezgif.com-video-to-gif-converter.gif" width="49%"/>
</p>

#### Demo Usage

We only support meshes in GLB format. If your mesh is not in GLB format, please convert it first. For our demo, you can simply drag a mesh into the browser, and the backend will encode the point cloud. Once encoding is complete, the camera view will adjust for optimal viewing. Then, you can interact with the demo by clicking on the mesh surface.

The performance of the Hugging Face demo is limited by data transfer and a shared GPU backend. To experience better performance, try running our local demo on your device by executing `python app.py`.

### Acknowledgement
We refer to [SAM](https://github.com/facebookresearch/segment-anything), [Uni3D](https://github.com/baaivision/Uni3D) and [OpenShape](https://github.com/Colin97/OpenShape_code) while developing Point-SAM. Thanks for these awesome works. We also thank @YouJiacheng for providing constructive suggestions for the data engine design.

### Citation
```
@misc{zhou2024pointsampromptable3dsegmentation,
      title={Point-SAM: Promptable 3D Segmentation Model for Point Clouds}, 
      author={Yuchen Zhou and Jiayuan Gu and Tung Yen Chiang and Fanbo Xiang and Hao Su},
      year={2024},
      eprint={2406.17741},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      url={https://arxiv.org/abs/2406.17741}, 
}
```

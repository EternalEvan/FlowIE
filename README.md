# FlowIE: Efficient Image Enhancement via Rectified Flow (CVPR 2024)

> [Yixuan Zhu](https://eternalevan.github.io/)\*, [Wenliang Zhao](https://wl-zhao.github.io/)\* $\dagger$, [Ao Li](https://rammusleo.github.io/), [Yansong Tang](https://andytang15.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/) $\ddagger$
> 
> \* Equal contribution &nbsp; $\dagger$ Project leader &nbsp; $\ddagger$ Corresponding author

The repository contains the official implementation for the paper "FlowIE: Efficient Image Enhancement via Rectified Flow" (CVPR 2024).

FlowIE is a simple yet highly effective <ins>**Flow**</ins>-based <ins>**I**</ins>mage <ins>**E**</ins>nhancement framework that estimates straight-line paths from an elementary distribution to high-quality images.
## 📋 To-Do List

* [ ] Release model and inference code.
* [ ] Release code for training dataloader .


## 💡 Pipeline

![](./assets/pipeline.png)

## ⭐️ Performance
<!--
![](./assets/performance.png)
![](./assets/table.png)

## 😀Quick Start
### ⚙️ 1. Installation

We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment. If you have installed Anaconda, run the following commands to create and activate a virtual environment.
``` bash
conda env create -f environment.yaml
conda activate dpmesh
```
### 💾 2. Data Preparation

We prepare the data in a samilar way as [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE) & [JOTR](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md). Please refer to [here](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md) for *dataset*, *SMPL model*, *VPoser model*. 

For 3DPW-OC and 3DPW-PC, we apply the same input key-points annotations as [JOTR](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md). Please refer to [3DPW-OC](https://drive.google.com/file/d/1IPE8Yw7ysd97Uv6Uw24el1yRs2r_HtCR/view?usp=sharing) & [3DPW-PC](https://drive.google.com/file/d/1xzZvUj1lR1ECbzUI4JOooC_r2LF6Qs5m/view?usp=sharing).

**For evaluation only, you can just prepare 3DPW dataset.**

```
|-- common
|   |-- utils
|   |   |-- human_model_files
|   |   |-- smplpytorch
|-- data 
|   |-- J_regressor_extra.npy 
|   |-- 3DPW
|   |   |-- 3DPW_latest_test.json
|   |   |-- 3DPW_oc.json
|   |   |-- 3DPW_pc.json
|   |   |-- 3DPW_validation_crowd_hhrnet_result.json
|   |   |-- imageFiles
|   |   |-- sequenceFiles
```


### 🗂️ 3. Download Checkpoints

Please download our pretrained checkpoints from [this link](https://cloud.tsinghua.edu.cn/d/1d6cd3ee30204bb59fce/) and put them under `./checkpoints`. The file directory should be:

```
|-- checkpoints
|--|-- 3dpw_best_ckpt.pth.tar
|--|-- 3dpw-crowd_best_ckpt.pth.tar
|--|-- 3dpw-oc_best_ckpt.pth.tar
|--|-- 3dpw-pc_best_ckpt.pth.tar
```

### 📊 4. Test & Evaluation

You can test DPMesh use following commands:

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun \
--master_port 29591 \
--nproc_per_node 1 \
eval.py \
--cfg ./configs/main_train.yml \
--exp_id="main_train" \
--distributed \
```

The evaluation process can be done with one Nvidia GeForce RTX 4090 GPU (24GB VRAM). You can use more GPUs by specifying the GPU ids.

## 🫰 Acknowledgments

We would like to express our sincere thanks to the author of [JOTR](https://github.com/xljh0520/JOTR) for the clear code base and quick response to our issues. 

We also thank [ControlNet](https://github.com/lllyasviel/ControlNet), [VPD](https://github.com/wl-zhao/VPD) and [LoRA](https://github.com/cloneofsimo/lora), for our code is partially borrowing from them.
-->
## 🔖 Citation

## 🔑 License

This code is distributed under an [MIT LICENSE](./LICENSE).

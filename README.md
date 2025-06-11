[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SdXSjEmH)
# EV-HW3: PhysGaussian

This homework is based on the recent CVPR 2024 paper [PhysGaussian](https://github.com/XPandora/PhysGaussian/tree/main), which introduces a novel framework that integrates physical constraints into 3D Gaussian representations for modeling generative dynamics.
<!-- 
You are **not required** to implement training from scratch. Instead, your task is to set up the environment as specified in the official repository and run the simulation scripts to observe and analyze the results.
 -->

## Environment Setup
```bash
git clone --recurse-submodules git@github.com:XPandora/PhysGaussian.git

cd PhysGaussian
conda create -n PhysGaussian python=3.9
conda activate PhysGaussian

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
apt install ffmpeg # If not downloaded

# Download the checkpoints provided from the original repo
pip install gdown
# bash download_sample_model.sh
gdown 17viNGxkhbJSlxtgJDK2sW2DEH3NdRwa-
unzip plane-trained.zip
mkdir model
mv plane-trained model/
```

### Troubleshooting

If you encounter issues during setup or execution, consider the following tips:

- **CUDA Errors:**  
    If you see errors related to CUDA header running on a WSL2 environment, try this:
    ```bash
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    ```
    You can refer to [this issue](https://github.com/taichi-dev/taichi/issues/8300) for more information.

- **Encoding Issues:**  
    If you get errors like `ascii codec can't encode character`, set the environment variable to use UTF-8 encoding: ([ref](https://stackoverflow.com/questions/56104377/ascii-codec-cant-encode-character-ordinal-not-in-range-128))
    ```bash
    export PYTHONIOENCODING=utf8
    ```
- **assertion failure: prealloc_size <= total_mem**
    Navigate to `gs_simulation.py` and change line 43
    ```python
    ti.init(arch=ti.cuda, device_memory_GB=8.0)
    ```
    to the suitable size for your GPU.


## Running the Simulation
<!-- Follow the "Quick Start" section and execute the simulation scripts as instructed. Make sure to verify your outputs and understand the role of physics constraints in the generated dynamics. -->
Usage:
```bash
cd PhysGaussian
CUDA_VISIBLE_DEVICES=0 python gs_simulation.py \
    --model_path <path/to/your/model> \
    --output_path <output/dir> \
    --config ./config/<your/config>.json \
    --render_img --compile_video --white_bg \
    --material <material>
```
- `--material`: Supported materials include `jelly`, `metal`, `sand`, `foam`, `snow` and `plasticine`

For example:
```bash
cd PhysGaussian
CUDA_VISIBLE_DEVICES=0 python gs_simulation.py \
    --model_path ./model/plane-trained/ \
    --output_path plane_metal \
    --config ./config/plane_config.json \
    --render_img --compile_video --white_bg \
    --material metal
```


<!-- ## Homework Instructions
Please complete Part 1–2 as described in the [Google Slides](https://docs.google.com/presentation/d/13JcQC12pI8Wb9ZuaVV400HVZr9eUeZvf7gB7Le8FRV4/edit?usp=sharing). -->

## Part 1

## Part 2


## Reference
```bibtex
@inproceedings{xie2024physgaussian,
    title     = {Physgaussian: Physics-integrated 3d gaussians for generative dynamics},
    author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```

## Appendix
Environment setup On meow1/2 server: (Thanks to classmate 陳仲肯 on Discord)
```bash
conda create -n PhysGaussian python=3.9
conda activate PhysGaussian

pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
conda install cuda-nvcc=11.8* cuda-libraries-dev=11.8* -c nvidia
conda install gxx=11.4.0 -c conda-forge

export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/ --verbose
pip install -e gaussian-splatting/submodules/simple-knn/ --verbose
```
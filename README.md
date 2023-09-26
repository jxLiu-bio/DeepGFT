# DeepGFT: clustering spatial transcriptomics data using deep learning and graph Fourier transform


<img src="https://img.shields.io/badge/Platform-Linux-green"> <img src="https://img.shields.io/badge/Language-python3-green"> <img src="https://img.shields.io/badge/License-MIT-green"> <img src="https://img.shields.io/badge/notebooks-passing-green">

## System Requirments

### OS requirements

```DeepGFT``` can run on Linux and Windows. The package has been tested on the following systems:

- Linux: CentOS 7
- Windows: Windows 10

```DeepGFT``` requires python version >= 3.7. We tested it in python 3.8.16 and cuda 11.6.1 on Linux.


## Installation Guide

### Create a virtual environment

Users can install ```anaconda``` by following this tutorial if there is no [Anaconda](https://www.anaconda.com/).

Create a separated virtual environment:

```shell
conda create -n DeepGFT python=3.8
conda activate DeepGFT
```


### Install packages

Install r-base and mclust packages:

```bash
conda install -c conda-forge r=4.1.0
conda install -c conda-forge r-mclust
```

Install ```DeepGFT``` from [Github](https://github.com/jxLiu-bio/DeepGFT) and [rpy2](https://pypi.org/project/rpy2/).

```bash
git clone https://github.com/jxLiu-bio/DeepGFT.git
cd DeepGFT
pip install -r requirement.txt
pip install rpy2==3.5.10
```

Next, run
```bash
python setup.py install
```

Install ```pytorch``` package of GPU version and ```pyG```.  See [Pytorch](https://pytorch.org/) and 
[PyG](https://pytorch-geometric.readthedocs.io/en/2.1.0/index.html) and for detail.
We passed the test on cuda 11.6.1. Users can choose the corresponding pytorch for other cuda versions. _torch_sparse_,
_torch_scatter_, _torch_cluster_ need to be manually downloaded on the [pytorch-geometric](https://pytorch-geometric.com/whl/).

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_sparse-0.6.16+pt112cu116-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.1.0+pt112cu116-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.6.0+pt112cu116-cp38-cp38-linux_x86_64.whl
pip install torch_geometric==2.1.0
```

Install ```jupyter notebook``` and set ipykernel.

```bash
conda install jupyter
python -m ipykernel install --user --name DeepGFT --display-name DeepGFT
```
## Tutorial
For the step-by-step tutorial, please refer to: [DeepGFT tutorial](https://deepgft.readthedocs.io/en/latest/notebook/2_Mouse_Primary_Visual_Cortex.html)

## Data availability
All data can be downloaded from https://drive.google.com/drive/folders/1uzrXJXbtwFomQuEldagfyA0Z_wfNqEza?usp=sharing.

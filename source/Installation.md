## Installation

### Create a virtual environment

Users can install ```anaconda``` by following this tutorial if there is no Conda. [https://www.anaconda.com/]

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

Install ```DeepGFT``` from Github and rpy2. See https://pypi.org/project/rpy2/ for detail.

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

Install ```pytorch``` package of GPU version and ```pyG```.  See https://pytorch.org/ and 
https://pytorch-geometric.readthedocs.io/en/2.1.0/index.html and for detail.
We passed the test on cuda 11.6.1. Users can choose the corresponding pytorch for other cuda versions. _torch_sparse_,
_torch_scatter_, _torch_cluster_ need to be manually downloaded on the https://pytorch-geometric.com/whl/.

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_sparse-0.6.16+pt112cu116-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.1.0+pt112cu116-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.6.0+pt112cu116-cp38-cp38-linux_x86_64.whl
pip install torch_geometric==2.1.0
```

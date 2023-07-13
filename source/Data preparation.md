## Data preparation

### Spatial transcriptomics data

We used four spatial gene expression datasets. 

1. The first dataset is the LIBD human dorsolateral prefrontal cortex (DLPFC) with 12 tissue slices from the package spatialLIBD (http://research.libd.org/spatialLIBD/). 

2. Mouse visual cortex STARmap data is obtained from the website (https://www.starmapresources.com/data). 

3. Two breast cancer datasets can be obtained from 10x Genomics Data Repository (https://www.10xgenomics.com/cn/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0 and https://www.10xgenomics.com/cn/resources/datasets/human-breast-cancer-block-a-section-2-1-standard-1-1-0). 

4. Human Lymph Node can be obtained from 10x Genomics Data Repository (https://www.10xgenomics.com/cn/resources/datasets/human-lymph-node-1-standard-1-1-0).

All data can be downloaded from https://drive.google.com/drive/folders/1uzrXJXbtwFomQuEldagfyA0Z_wfNqEza?usp=sharing.

We recommend load Visium data by:
```
import scanpy as sc
adata = sc.read_visium(path_to_visium_dataset)
```
For all spatial transcriptomics datasets, it should be pointed out that raw count matrix needs to be found at _adata.X_ and the spatial coordinate information needs to be found at _adata.obs_ or _adata.obsm_



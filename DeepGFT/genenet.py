import mygene
import numpy as np
from scipy import io
import pandas as pd

mg = mygene.MyGeneInfo()


def obtain_genenet(adata, dataset='pearson', species='human', path='', percentile=0, cut=0.5):
    """
    Constructing Gene Networks from Gene Pools or Pearson Coefficients.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        dataset: str, optional
            Choose a gene bank or Pearson. The default is 'pearson'.
        species: str, optional
            If selecting a gene pool, select a species. The default is 'human'.
        path: str, optional
            Gene pool pathway. The default is ''.
        percentile: int, optional
            Select the quantile to determine whether to use sparsity to find the threshold. The default is 0.

    Returns:
        None
    """
    if dataset == 'COEXPEDIA':
        adata.var_names = [gene.upper() for gene in adata.var_names]
        gene_list = np.sort(adata.var_names)
        network = io.mmread(path + 'genenet/COEXPEDIA/coexpedia_network_' + species + '.mtx').toarray()
        netgene = np.load(path + 'genenet/COEXPEDIA/coexpedia_gene_names_' + species + '_symbol.npy')
        genes = [gene for gene in netgene]
        data_idtype = get_geneid_type(genes)
        if data_idtype != 'symbol':
            genes = id2symbol(genes, specise=species)
        else:
            genes = genes
        genes = [gene.upper() for gene in genes]
        result = pd.DataFrame(network, index=genes, columns=genes)
        common_genes = np.intersect1d(genes, gene_list)
        result = result.loc[common_genes, common_genes]

        net = result

        adata.uns['genenet_cutoff'] = genenet_cutoff(net)
        adata.uns['genenet'] = net

    elif dataset == 'pearson':
        if type(adata.X) == np.ndarray:
            X = adata.X.copy()
        else:
            X = adata.X.copy().todense()
        df = pd.DataFrame(X)
        if adata.shape[0] > 5000:
            spot_choose = np.random.choice(list(range(adata.shape[0])), 5000, replace=False)
            net_choose = df.loc[spot_choose, :]
            genenet = net_choose.corr(method='pearson')
        else:
            genenet = df.corr(method='pearson')
        if percentile != 0:
            genenet_num = genenet.values.reshape(-1)
            genenet_percentile = np.percentile(genenet_num, percentile)
            adata.uns['genenet'] = genenet * (genenet > genenet_percentile)
        else:
            adata.uns['genenet'] = genenet * (genenet >= cut)
        print('gene edges',(adata.uns['genenet'].values > 0).sum().sum(), 'spots', adata.shape[0])

    # elif dataset == 'COXPRESdb':
    #     a = pd.read_csv('genenet/COXPRESdb/Hsa-m2.c4-0.expression.combat.txt', delimiter='\t')
    #     b = pd.read_csv('genenet/COXPRESdb/Hsa-m.c7-0.expression.combat.txt', delimiter='\t')
    #     for i in range(a.shape[0]):
    #         if a[i] == '':
    #             print(i)


def genenet_cutoff(net):
    max_value = net.max().max()
    net = net > 0
    net = net.astype(int)
    return net


def id2symbol(geneid, specise='human'):
    geneSyms = [get_gene(x) for x in
                mg.querymany(geneid, scopes=["ensemblgene", "entrezgene"], fields='symbol', species=specise)]
    return geneSyms


def get_geneid_type(genes):
    if genes[0][0:3] == 'ENS' or genes[0][0:3] == 'ens':
        return 'ensemble'
    if RepresentsInt(genes[0]):
        return 'entrez'
    return 'symbol'


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_gene(x):
    if 'symbol' in x.keys():
        return x['symbol']
    return ''

import argparse
import pandas as pd
import numpy as np
import sklearn
import torch
import random
import os
import torch_geometric
import scipy.sparse as sp
from torch_geometric.data import Data
from numpy.linalg import norm
from DeepGFT.svg_select import *
from scipy.sparse import csr_matrix
import scipy.sparse as ss
import ot


def parameter_setting_spot():
    parser = argparse.ArgumentParser(description='Spatial transcriptomics analysis')
    parser.add_argument('--name', type=str, default='151673',
                        help='Name for the input of datasets')
    parser.add_argument('--n_svg', type=int, default=3000,
                        help='Number of spatially variable genes selected')
    parser.add_argument('--signal', '-s', type=int, default=1500,
                        help='Signal quantity of spots and genes')
    parser.add_argument('--csvg', type=float, default=1e-4,
                        help='Smoothing coefficient of GFT for noise reduction')
    parser.add_argument('--c', type=float, default=0.05,
                        help='Smoothing coefficient of GFT for training')
    parser.add_argument('--middle', type=int, default=3,
                        help='The number of filtered signal values based on the eigenvalues')
    parser.add_argument('--alpha', type=float, default=20,
                        help='Proportion of loss function in spatial domain and spectral domain')
    parser.add_argument('--epoch', type=int, default=600,
                        help='Maximum number of training sessions')

    return parser


def array2torch(x, device='cuda:0'):
    """Convert elements in x into tensors"""
    Tensor = []
    for i in x:
        Tensor.append(torch.Tensor(i).to(device))
    return Tensor


def torch2array(x):
    """Convert elements in x into arrays"""
    array = []
    for i in x:
        array.append(i.cpu().detach().numpy())
    return array


def csr(adata):
    """Convert the expression matrix to a CSR matrix"""
    row, col = np.nonzero(adata.X)
    values = adata.X[row, col]
    csr_x = csr_matrix((values, (row, col)), shape=(adata.shape[0], adata.shape[1]))
    adata.X = csr_x


def denoising(X, adj, att):
    """
    Using attention coefficient to reduce noise reconstructed matrix.

    .. math::
        \widetilde{X} = X+\frac{c}{\lambda_{max}}AX.

    Args:
        X: array
            Gene expression matrix with n spots and m genes.
        adj: array
            Adjacency matrix with n points.
        att: array
            Calculated attention coefficient.

    Returns:
        csr_X: csr_matrix
            Gene expression matrix after noise reduction.
        csr_adj: csr_matrix
            Attention matrix after noise reduction.
    """

    adj = adj + np.eye(adj.shape[0])
    row, col = np.nonzero(adj.values)
    csr_adj = csr_matrix((att, (row, col)), shape=(adj.shape[0], adj.shape[0]))
    lap_mtx = _get_lap_mtx(csr_adj)
    eigvals, _ = ss.linalg.eigsh(lap_mtx.astype('double'), k=1, which='LM')
    lambda_max = eigvals[0]
    Y = np.array(0.1 / lambda_max * np.matmul(csr_adj.toarray(), np.exp(X)))
    Y = np.log1p(Y)

    row, col = np.nonzero(Y)
    values = Y[row, col]
    csr_X = csr_matrix((values, (row, col)), shape=(X.shape[0], X.shape[1]))
    return csr_X, csr_adj


def refine_label(adata, radius=50, key='label'):
    """Used for smoothing clustering results"""
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)
    return new_type


def feature2signal(adata, gene_signal=1024, spot_signal=1024, c1=0.05, c2=0.0001, middle=10):
    """
    Transform spatial domain features into spectral domain signals using Graph Fourier transform

    Args:
        adata: anndata
            AnnData object of scanpy package.
        gene_signal: int, optional
            The number of gene signals. The default is 1024.
        spot_signal: int, optional
            The number of spot signals. The default is 1024.
        c1: float, optional
            Filter coefficients of gene signals. The default is 0.05.
        c2: float, optional
            Filter coefficients of spot signals. The default is 0.0001.
        middle: int, optional
            The number of filtered signal values based on the eigenvalues. The default is 10.

    Returns:
        gene_freq_mtx.T:
            Matrix of genes multiplied by signals.
        gene_eigvecs_T:
            Matrix of gene feature vectors.
        gene_eigvals:
            Vector of eigenvalues of genes.
        spot_freq_mtx.T:
            Matrix of spots multiplied by signals.
        spot_eigvecs_T:
            Matrix of spot feature vectors.
        spot_eigvals:
             Vector of eigenvalues of spots.
    """
    spot_net = adata.uns['spotnet_adj']
    if spot_net.shape[0] >= 2e4:
        gene_freq_mtx = np.matmul(0.7 * np.eye(spot_net.shape[0]) + 0.3 * spot_net, adata.X.copy().todense())
        gene_eigvecs_T = np.eye(spot_net.shape[0])
        gene_eigvals = None
    else:
        gene_freq_mtx, gene_eigvecs_T, gene_eigvals = GraphFourierTransform(spot_net, adata.X.copy().todense(),
                                                                            n_GFT=gene_signal)
        gene_freq_mtx = np.matmul(np.diag([1 / (1 + c1 * eig) for eig in gene_eigvals]), gene_freq_mtx)
        # gene_freq_mtx = np.matmul(np.diag([np.exp(- eig / 200) for eig in gene_eigvals]), gene_freq_mtx)

    gene_net = adata.uns['genenet']
    spot_freq_mtx, spot_eigvecs_T, spot_eigvals = GraphFourierTransform(gene_net, adata.X.copy().todense().T,
                                                                        n_GFT=spot_signal,
                                                                        middle_pass=True,
                                                                        middle=middle)
    spot_freq_mtx = np.matmul(np.diag([1 / (1 + c2 * eig) for eig in spot_eigvals]), spot_freq_mtx)
    # spot_freq_mtx = np.matmul(np.diag([np.exp(- eig / 200) for eig in spot_eigvals]), spot_freq_mtx)
    adata.uns['signal'] = pd.DataFrame(spot_freq_mtx.T, index=adata.obs.index)
    return gene_freq_mtx.T, gene_eigvecs_T, gene_eigvals, spot_freq_mtx.T, spot_eigvecs_T, spot_eigvals


def f2s_gene(adata, gene_signal=1024, c1=0.05):
    """Using spot networks to obtain gene signals, eigenvectors, and eigenvalues through GFT"""
    spot_net = adata.uns['spotnet_adj']
    if spot_net.shape[0] >= 2e4:
        gene_freq_mtx = np.matmul(0.7 * np.eye(spot_net.shape[0]) + 0.3 * spot_net, adata.X.copy().todense())
        gene_eigvecs_T = np.eye(spot_net.shape[0])
        gene_eigvals = None
    else:
        gene_freq_mtx, gene_eigvecs_T, gene_eigvals = GraphFourierTransform(spot_net, adata.X.copy().todense(),
                                                                            n_GFT=gene_signal)
        gene_freq_mtx = np.matmul(np.diag([1 / (1 + c1 * eig) for eig in gene_eigvals]), gene_freq_mtx)
    adata.uns['gene_freq'] = gene_freq_mtx
    return gene_freq_mtx.T, gene_eigvecs_T, gene_eigvals


def f2s_spot(adata, spot_signal=1024, c2=0.05, middle=10):
    """Using gene networks to obtain spot signals, eigenvectors, and eigenvalues through GFT"""
    gene_net = adata.uns['genenet']
    spot_freq_mtx, spot_eigvecs_T, spot_eigvals = GraphFourierTransform(gene_net, adata.X.todense().T,
                                                                        n_GFT=spot_signal, middle_pass=True,
                                                                        middle=middle)
    spot_freq_mtx = np.matmul(np.diag([1 / (1 + c2 * eig) for eig in spot_eigvals]), spot_freq_mtx)
    adata.uns['signal'] = pd.DataFrame(spot_freq_mtx.T, index=adata.obs.index)
    return spot_freq_mtx.T, spot_eigvecs_T, spot_eigvals


def GraphFourierTransform(net, X, n_GFT=256, middle_pass=False, middle=5):
    """Using networks to obtain signals, eigenvectors, and eigenvalues through GFT"""
    # remove useless genes
    lap_mtx = _get_lap_mtx(net)
    # Obtain Fourier modes and corresponding eigenvalues
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx.astype('double'),
                                       k=n_GFT,
                                       which='SM',
                                       v0=[1 / np.sqrt(net.shape[0])] * net.shape[0])
    if middle_pass:
        eigvals = eigvals[middle:]
        eigvecs = eigvecs[:, middle:]
    eigvecs_T = eigvecs.transpose()

    if ss.isspmatrix(X):
        exp_mtx = X.todense()
    else:
        exp_mtx = X
    if middle_pass:
        # exp_mtx = sklearn.preprocessing.scale(np.array(exp_mtx), axis=1)
        freq_mtx = np.matmul(eigvecs_T, exp_mtx)
        # freq_mtx = sklearn.preprocessing.normalize(np.array(freq_mtx), norm='l2', axis=0)
    else:
        freq_mtx = np.matmul(eigvecs_T, exp_mtx)

    return freq_mtx, eigvecs_T, eigvals


def _create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)), shape=(diag.size, diag.size))
    return sparse_mtx


def _get_lap_mtx(gene_net):
    diag = gene_net.sum(axis=1)
    deg_mtx = _create_degree_mtx(diag)
    adj_mtx = ss.coo_matrix(gene_net)
    lap_mtx = deg_mtx - adj_mtx
    return lap_mtx


def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y']):
    """
    Subgraph training by partitioning points.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        num_batch_x: int, optional
            Number of divisions along the horizontal axis. The default is 2.
        num_batch_y: int, optional
            Number of divisions along the vertical axis. The default is 2.
        spatial_key: list, optional
            Column names for spatial locations. The default is ['X', 'Y'].

    Returns:
        Batch_list:
            List of divided annData object of scanpy package.
    """
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1 / num_batch_x) * x * 100) for x in range(num_batch_x + 1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1 / num_batch_y) * x * 100) for x in range(num_batch_y + 1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x + 1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y + 1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    return Batch_list


def Batch_Data_3D(adata, num_batch_x, num_batch_y, num_batch_z, spatial_key=['X', 'Y', 'Z']):
    """
    3D subgraph training by partitioning points.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        num_batch_x: int, optional
            Number of divisions along the horizontal axis. The default is 2.
        num_batch_y: int, optional
            Number of divisions along the vertical axis. The default is 2.
        num_batch_z: int, optional
            Number of divisions along the vertical axis. The default is 1.
        spatial_key: list, optional
            Column names for spatial locations. The default is ['X', 'Y', 'Z'].

    Returns:
        Batch_list:
            List of divided annData object of scanpy package.
    """
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1 / num_batch_x) * x * 100) for x in range(num_batch_x + 1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1 / num_batch_y) * x * 100) for x in range(num_batch_y + 1)]
    batch_z_coor = [np.percentile(Sp_df[:, 2], (1 / num_batch_z) * x * 100) for x in range(num_batch_z + 1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            for it_z in range(num_batch_z):
                min_x = batch_x_coor[it_x]
                max_x = batch_x_coor[it_x + 1]
                min_y = batch_y_coor[it_y]
                max_y = batch_y_coor[it_y + 1]
                min_z = batch_z_coor[it_z]
                max_z = batch_z_coor[it_z + 1]
                temp_adata = adata.copy()
                temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
                temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
                temp_adata = temp_adata[temp_adata.obs[spatial_key[2]].map(lambda z: min_z <= z <= max_z)]
                Batch_list.append(temp_adata)
    return Batch_list


def obtain_spotnet(adata, rad_cutoff=150, k_cutoff=6, knn_method='KNN', prune=True):
    """
    Constructing a graph of spots using KNN or Radius methods.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        rad_cutoff: int, optional
            Truncation radius of spots. The default is 150.
        k_cutoff: int, optional
            Number of adjacent spots of a spot. The default is 6.
        knn_method: str, optional
            The method of constructing a graph of points, including 'KNN' and 'Radius'. The default is 'KNN'.
        prune: bool, optional
            Determine whether to prune or not. The default is True.

    Returns:
        None
    """
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    # coor.columns = ['imagerow', 'imagecol']
    KNN_list = []
    if knn_method == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
    if knn_method == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        if prune:
            for l in range(indices.shape[0]):
                d = distances[l][np.argwhere(distances[l])]
                boundary = np.mean(d) + np.std(d)
                distances[l] *= (distances[l] <= boundary)
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    spot_net = csr_matrix((Spatial_Net['Distance'].values > 0, Spatial_Net[['Cell1', 'Cell2']].values.T)).toarray()
    spot_net = pd.DataFrame(spot_net.astype(int), index=adata.obs_names, columns=adata.obs_names)
    adata.uns['spotnet_adj'] = spot_net
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    adata.uns['spotnet'] = Spatial_Net


def obtain_pre_spotnet(adata, adata_raw, pre_feature=False, k_cutoff=6, res_pre=0.6):
    """
    Obtain a graph of spots through pre-clustering.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        adata_raw: anndata.
            Raw annData object of scanpy package.
        pre_feature: bool, optional
            Determine whether to use features to obtain a graph of spots. The default is False.
        k_cutoff: int, optional
            Number of adjacent points in constructing a graph using features. The default is 6.
        res_pre: float, optional
            Resolution value for clustering. The default is 1.0.

    Returns:
        None
    """
    # adata.obsm['emb'] = adata.uns['signal'].copy()
    # sc.pp.neighbors(adata, use_rep='emb')
    # raw = sc.AnnData(adata.uns['raw'].todence(), obs=adata.obs_names).
    raw = adata_raw.copy()
    sc.pp.pca(raw, svd_solver='arpack')
    sc.pp.neighbors(raw)
    sc.tl.louvain(raw, resolution=res_pre, key_added='expression_louvain_label')
    # new_type = refine_label(raw, radius=50, key='expression_louvain_label')
    # raw.obs['expression_louvain_label'] = new_type
    # sc.pl.spatial(raw, color=['expression_louvain_label'], title=[1], save='louvain' + '.png')
    pre_cluster = 'expression_louvain_label'
    prune_G_df = prune_spatial_Net(adata.uns['spotnet'].copy(), raw.obs[pre_cluster])
    adata.uns['spotnet_cluster'] = prune_G_df
    del adata.uns['spotnet_cluster']['Cell1_label']
    del adata.uns['spotnet_cluster']['Cell2_label']

    if pre_feature:
        feature = adata.uns['signal'].copy()
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            cos_sim = cosine(feature.iloc[indices[it, 1:]].values, feature.iloc[it].values)
            cos_sim = cos_sim * (cos_sim >= np.mean(cos_sim[1:]))
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], cos_sim)))
        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Cosine_similarity']
        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[Spatial_Net['Cosine_similarity'] > 0,]
        id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
        adata.uns['spotnet_feature'] = Spatial_Net


def cosine(A, B):
    return np.dot(A, B) / (norm(A, axis=1) * norm(B))


def Transfer_pytorch_Data(adata, device, graph=None):
    if graph is None:
        graph = ['none', 'cluster']
    data = []
    for i in graph:
        G_df = adata.uns[i].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
        G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

        G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
        G = G + sp.eye(G.shape[0])

        edgeList = np.nonzero(G)
        if type(adata.X) == np.ndarray:
            data.append(Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]]))).to(device))
        else:
            data.append(Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]]))).to(device))
    if len(data) == 1:
        if type(adata.X) == np.ndarray:
            data.append(Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]]))).to(device))
        else:
            data.append(Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]]))).to(device))
    return data


def prune_spatial_Net(Graph_df, label):
    pro_labels_dict = dict(zip(label.index, label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label'] == Graph_df['Cell2_label'],]
    return Graph_df


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None):
    """
    Constructing KNN Graph by Combining 2D and 3D graph.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        rad_cutoff_2D: int, optional
            The radius used for constructing KNN graphs in 2D graph. The default is 50.
        rad_cutoff_Zaxis: int, optional
            The radius used for constructing KNN graphs in 3D graph. The default is 50.
        key_section: str, optional
            Column names for slices. The default is 'Section_id'.
        section_order: str, default
            The name of the slice. The default is None.

    Returns:
        None
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    for temp_section in np.unique(adata.obs[key_section]):
        temp_adata = adata[adata.obs[key_section] == temp_section,]
        obtain_spotnet(temp_adata, knn_method='Radius', rad_cutoff=rad_cutoff_2D)
        temp_adata.uns['spotnet']['SNN'] = temp_section
        adata.uns['Spatial_Net_2D'] = pd.concat([adata.uns['Spatial_Net_2D'], temp_adata.uns['spotnet']])
    for it in range(num_section - 1):
        section_1 = section_order[it]
        section_2 = section_order[it + 1]
        Z_Net_ID = section_1 + '-' + section_2
        temp_adata = adata[adata.obs[key_section].isin([section_1, section_2]),]
        obtain_spotnet(temp_adata, knn_method='Radius', rad_cutoff=rad_cutoff_Zaxis)
        spot_section_trans = dict(zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['spotnet']['Section_id_1'] = temp_adata.uns['spotnet']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['spotnet']['Section_id_2'] = temp_adata.uns['spotnet']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['spotnet'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['spotnet'] = temp_adata.uns['spotnet'].loc[used_edge,]
        temp_adata.uns['spotnet'] = temp_adata.uns['spotnet'].loc[:, ['Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['spotnet']['SNN'] = Z_Net_ID
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['spotnet']])
    adata.uns['spotnet'] = pd.concat([adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])

    spotnet = adata.uns['spotnet'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    spotnet['Cell1'] = spotnet['Cell1'].map(cells_id_tran)
    spotnet['Cell2'] = spotnet['Cell2'].map(cells_id_tran)
    spot_net = csr_matrix((spotnet['Distance'].values > 0, spotnet[['Cell1', 'Cell2']].values.T)).toarray()
    spot_net = pd.DataFrame(spot_net.astype(int), index=adata.obs_names, columns=adata.obs_names)
    adata.uns['spotnet_adj'] = spot_net


def svg(adata, svg_method='gft_top', n_top=3000, csvg=0.0001, smoothing=True):
    """
    Select spatially variable genes using six methods, including 'gft', 'gft_top',
    'seurat', 'seurat_v3', 'cell_ranger' and 'mix'.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        svg_method: str, optional
            Methods for selecting spatially variable genes. Teh default is 'gft_top'.
        n_top: int, optional
            Number of spatially variable genes selected. The default is 3000.
        csvg: float, optional
            Smoothing coefficient of GFT for noise reduction. The default is 0.0001.
        smoothing: bool, optional
            Determine whether it is smooth for noise reduction. The default is True.

    Returns:
        adata: anndata
            AnnData object of scanpy package after choosing svgs and smoothing.
        adata_raw: anndata
            AnnData object of scanpy package before choosing svgs and smoothing.
    """
    assert svg_method in ['gft', 'gft_top', 'seurat', 'seurat_v3', 'cell_ranger', 'mix']
    if svg_method == 'seurat_v3':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
        adata = adata[:, adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    elif svg_method == 'mix':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=int(n_top / 2))
        seuv3_list = adata.var_names[adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gene_df = rank_gene_smooth(adata,
                                   spatial_info=['array_row', 'array_col'],
                                   ratio_low_freq=1,
                                   ratio_high_freq=1,
                                   ratio_neighbors=1,
                                   filter_peaks=True,
                                   S=6)
        svg_list = gene_df.index[:(n_top - len(seuv3_list))]
        merged_gene_list = np.union1d(seuv3_list, svg_list)
        adata = adata[:, merged_gene_list]
        if smoothing:
            low_pass_enhancement(adata,
                                 ratio_low_freq=15,
                                 c=csvg,
                                 spatial_info='spatial',
                                 ratio_neighbors=0.3,
                                 inplace=True)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if svg_method == 'seurat':
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
        elif svg_method == 'cell_ranger':
            sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
        elif svg_method == 'gft' or svg_method == 'gft_top':
            gene_df = rank_gene_smooth(adata,
                                       spatial_info='spatial',
                                       ratio_low_freq=1,
                                       ratio_high_freq=1,
                                       ratio_neighbors=1,
                                       filter_peaks=True,
                                       S=6)
            if svg_method == 'gft':
                svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < 0.05].index.tolist()
            elif svg_method == 'gft_top':
                svg_list = gene_df.index[:n_top].tolist()
            adata = adata[:, svg_list]
            adata_raw = adata.copy()
            if smoothing:
                low_pass_enhancement(adata,
                                     ratio_low_freq=15,
                                     c=csvg,
                                     spatial_info='spatial',
                                     ratio_neighbors=0.3,
                                     inplace=True)
    return adata, adata_raw


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=3, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2023):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library('mclust')

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def seed_all(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch_geometric.seed.seed_everything(2023)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch_sparse
from DeepGFT.DeepGFT import *
import torch
from tqdm import tqdm
from DeepGFT.utils import array2torch, torch2array, Transfer_pytorch_Data


def train_spot(adata, gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T,
               epoch_max=600, alpha=10, device='cuda:0', lr=0.001, hidden_dims=[256, 30], cluster=True):
    """
    Using autoencoder for training to obtain low dimensional features of points.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        gene_freq_mtx: array
            Matrix of genes multiplied by signals.
        gene_eigvecs_T: array
            The Eigenvector Matrix of genes.
        spot_freq_mtx: array
            Matrix of spots multiplied by signals.
        spot_eigvecs_T: array
             The Eigenvector Matrix of spots.
        epoch_max: int, optional
            Maximum training times. The default is 600.
        alpha: float, optional
            Proportion of loss function in spatial domain and spectral domain. The default is 10.
        device: str, optional
            Training equipment. The default is 'cuda:0'.
        lr: float, optional
            Learning rate. The default is 0.001.
        hidden_dims: list, optional
            Number of neurons in the hidden and intermediate layers. The default is [256, 30].
        cluster: bool, optional
            Whether to use pre clustering function. The default is True.

    Returns:
        res: array
            Reconstructed gene expression matrix.
        lamda:
            Fusion ratio of spot signal and gene signal.
        emb_spot:
            Low dimensional embedding of spots.
        mtx_res_spot:
            Signal matrix for spot reconstruction
        attention:
            Attention coefficient output by autoencoder.
    """
    print('spot*signal train')
    X = adata.X.copy()
    graph = ['spotnet', 'spotnet_cluster'] if cluster else ['spotnet']
    data = Transfer_pytorch_Data(adata, device, graph)
    data[0].edge_index = torch_sparse.SparseTensor(row=data[0].edge_index[0], col=data[0].edge_index[1])
    data[1].edge_index = torch_sparse.SparseTensor(row=data[1].edge_index[0], col=data[1].edge_index[1])
    model = SPOT(dim=X.shape[1], dims_spot=[spot_freq_mtx.shape[1]] + hidden_dims, device=device, cluster=cluster).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    [gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T, origin] \
        = array2torch([gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T, X.todense()], device)
    train_loss_list = []
    for epoch in tqdm(range(epoch_max)):
        model.train()
        optimizer.zero_grad()
        res, _, _, mtx_res_spot = \
            model(gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T,
                  data[0].edge_index, data[1].edge_index)
        loss = F.mse_loss(res, origin) + alpha * F.mse_loss(mtx_res_spot, spot_freq_mtx)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss)

        # if len(train_loss_list) >= 2:
        #     if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-20:
        #         print("converged!!!")
        #         break
    model.eval()
    res, lamda, emb_spot, mtx_res_spot = \
        model(gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T,
              data[0].edge_index, data[1].edge_index)
    try:
        attention = model.GATE_spot.conv1.attention
    except:
        attention = torch.Tensor([1]).to(device)
    [res, lamda, emb_spot, mtx_res_spot, attention] = torch2array([res, lamda, emb_spot, mtx_res_spot, attention])
    return res, lamda, emb_spot, mtx_res_spot, attention


def train_spot_batch(adata, Batch_list,
                     gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T,
                     epoch_max=600, alpha=10, device='cuda:0', lr=0.001, hidden_dims=[256, 30], cluster=True):
    """
    Use automatic encoders to train subgraphs in batches to obtain low dimensional features of spots.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        Batch_list: list
            List of divided annData object of scanpy package.
        gene_freq_mtx: array
            Matrix of genes multiplied by signals.
        gene_eigvecs_T: array
            The Eigenvector Matrix of genes.
        spot_freq_mtx: array
            Matrix of spots multiplied by signals.
        spot_eigvecs_T: array
             The Eigenvector Matrix of spots.
        epoch_max: int, optional
            Maximum training times. The default is 600.
        alpha: float, optional
            Proportion of loss function in spatial domain and spectral domain. The default is 10.
        device: str, optional
            Training equipment. The default is 'cuda:0'.
        lr: float, optional
            Learning rate. The default is 0.001.
        hidden_dims: list, optional
            Number of neurons in the hidden and intermediate layers. The default is [256, 30].
        cluster: bool, optional
            Whether to use pre clustering function. The default is True.

    Returns:
        res: array
            Reconstructed gene expression matrix.
        lamda:
            Fusion ratio of spot signal and gene signal.
        emb_spot:
            Low dimensional embedding of spots.
        mtx_res_spot:
            Signal matrix for spot reconstruction
        attention:
            Attention coefficient output by autoencoder.
    """
    print('spot*signal train')
    spot_freq_mtx = np.array(spot_freq_mtx)
    gene_freq_mtx = np.array(gene_freq_mtx)
    spot_freq_pd = pd.DataFrame(spot_freq_mtx, index=adata.obs_names)
    gene_eigvecs_T_pd = pd.DataFrame(gene_eigvecs_T, columns=adata.obs_names)
    graph = ['spotnet', 'spotnet_cluster'] if cluster else ['spotnet']
    data_all = Transfer_pytorch_Data(adata, device, graph)
    data_all[0].edge_index = torch_sparse.SparseTensor(row=data_all[0].edge_index[0], col=data_all[0].edge_index[1])
    data_all[1].edge_index = torch_sparse.SparseTensor(row=data_all[1].edge_index[0], col=data_all[1].edge_index[1])
    [gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T] \
        = array2torch([gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T], device)

    # Batch training
    spotnet_list, origin_list, spot_freq_list, gene_eigvecs_T_list = [], [], [], []
    spotnet_all = adata.uns['spotnet'].copy()
    for adata_tmp in Batch_list:
        mask = np.isin(spotnet_all['Cell1'].values, adata_tmp.obs_names) * np.isin(spotnet_all['Cell2'].values, adata_tmp.obs_names)
        adata_tmp.uns['spotnet'] = spotnet_all[mask]
        if cluster:
            spotnet_cluster_all = adata.uns['spotnet_cluster'].copy()
            mask_cluster = np.isin(spotnet_cluster_all['Cell1'].values, adata_tmp.obs_names) * np.isin(spotnet_cluster_all['Cell2'].values, adata_tmp.obs_names)
            adata_tmp.uns['spotnet_cluster'] = spotnet_cluster_all[mask_cluster]
        spotnet_list.append(Transfer_pytorch_Data(adata_tmp, device, graph))
        origin_list.append(array2torch([adata_tmp.X.copy().todense()], device))
        spot_freq_list.append(array2torch([spot_freq_pd.loc[adata_tmp.obs_names, :].values], device))
        gene_eigvecs_T_list.append(array2torch([gene_eigvecs_T_pd.loc[:, adata_tmp.obs_names].values], device))

    for data in spotnet_list:
        data[0].edge_index = torch_sparse.SparseTensor(row=data[0].edge_index[0], col=data[0].edge_index[1])
        data[1].edge_index = torch_sparse.SparseTensor(row=data[1].edge_index[0], col=data[1].edge_index[1])

    model = SPOT(dim=adata.X.copy().shape[1], dims_spot=[spot_freq_mtx.shape[1]] + hidden_dims, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    train_loss_list = []
    for epoch in tqdm(range(epoch_max)):
        for batch in range(len(Batch_list)):
            model.train()
            optimizer.zero_grad()
            res, _, _, mtx_res_spot = \
                model(gene_freq_mtx, gene_eigvecs_T_list[batch][0], spot_freq_list[batch][0], spot_eigvecs_T,
                      spotnet_list[batch][0].edge_index, spotnet_list[batch][1].edge_index)
            loss = F.mse_loss(res, origin_list[batch][0]) + alpha * F.mse_loss(mtx_res_spot, spot_freq_list[batch][0])
            loss.backward()
            optimizer.step()
            # train_loss_list.append(loss)

            # if len(train_loss_list) >= 2:
            #     if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-20:
            #         print("converged!!!")
            #         break

    model.eval()
    model.to('cpu')
    res, lamda, emb_spot, mtx_res_spot = \
        model(gene_freq_mtx.to('cpu'), gene_eigvecs_T.to('cpu'), spot_freq_mtx.to('cpu'), spot_eigvecs_T.to('cpu'),
              data_all[0].edge_index.to('cpu'), data_all[1].edge_index.to('cpu'))
    try:
        attention = model.GATE_spot.conv1.attention
    except:
        attention = torch.Tensor([1]).to(device)
    # [res, lamda, emb_spot, mtx_res_spot] = torch2array([res, lamda, emb_spot, mtx_res_spot])

    return res, lamda, emb_spot, mtx_res_spot, attention


def train_gene(gene_freq_mtx, genenet, genenet_cutoff, device='cuda:0', lr=0.001, hidden_dims=[64, 16],
               dropout=0.02, weight=0.00001, epoch_max=600, gamma=1):
    print('gene*signal train')
    genenet = genenet.values
    genenet = genenet / genenet.max().max()
    genenet_cutoff = genenet_cutoff.values
    model = GENE(dims_gene=[gene_freq_mtx.shape[1]] + hidden_dims, dropout=dropout, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight)
    loss_tmp = np.inf

    [gene_freq_mtx, genenet, genenet_cutoff] = array2torch([gene_freq_mtx, genenet, genenet_cutoff], device)
    # genenet = F.sigmoid(genenet)
    train_loss_list = []
    # for epoch in tqdm(range(epoch_max)):
    for epoch in range(epoch_max):
        model.train()
        optimizer.zero_grad()
        _, genenet_res, _ = model(gene_freq_mtx, genenet_cutoff)
        loss1 = F.binary_cross_entropy_with_logits(genenet_res, genenet)
        # loss1 = F.mse_loss(genenet_res, genenet_cutoff, reduction='sum')
        loss2 = torch.sqrt(F.mse_loss((genenet_res > 0).sum(axis=1), genenet_cutoff.sum(axis=1)) / genenet.shape[0])
        loss = loss1 + gamma * loss2
        print(loss1.item(), loss2.item())
        print('epoch{}: '.format(epoch), loss.item())
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss)

        # if len(train_loss_list) >= 2:
        #     if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-7:
        #         print("converged!!!")
        #         break
    model.eval()
    emb_gene, genenet_res, attention = model(gene_freq_mtx, genenet_cutoff)
    [emb_gene, genenet_res, attention] = torch2array([emb_gene, genenet_res, attention])
    return emb_gene, genenet_res, attention

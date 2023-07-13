import torch
import torch.nn as nn
from DeepGFT.layers import *
from torch.nn.modules.module import Module
from DeepGFT.layers import GraphAttentionLayer
import torch.nn.functional as F
from DeepGFT.gatv2_conv import GATv2Conv
from torch_geometric.nn import GATConv, TransformerConv


class GAT(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GAT, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv1_p = GATv2Conv(in_dim, num_hidden, heads=3, concat=False,
                                 dropout=0.2, add_self_loops=False, bias=True)

        self.conv2 = torch.nn.Linear(num_hidden * 2, out_dim, bias=True)

        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv3_p = GATv2Conv(out_dim, num_hidden, heads=3, concat=False,
                                 dropout=0.2, add_self_loops=False, bias=True)

        self.conv4 = torch.nn.Linear(num_hidden * 2, in_dim, bias=True)

    def forward(self, features, edge_index, edge_index_cluster):

        h1_nonprune = self.conv1(features, edge_index)
        h1_prune = self.conv1_p(features, edge_index_cluster)
        h1 = F.elu(torch.cat([h1_nonprune, h1_prune], dim=1))

        h2 = self.conv2(h1)

        h3_nonprune = self.conv3(h2, edge_index)
        h3_prune = self.conv3_p(h2, edge_index_cluster)
        h3 = F.elu(torch.cat([h3_nonprune, h3_prune], dim=1))

        h4 = self.conv4(h3)

        return h2, h4  #, self.conv3._alpha


class GAT_noncluster(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GAT_noncluster, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        num_hidden *= 2
        self.conv1 = GATv2Conv(in_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv2 = torch.nn.Linear(num_hidden, out_dim, bias=True)
        self.conv3 = GATv2Conv(out_dim, num_hidden, heads=2, concat=False,
                               dropout=0.2, add_self_loops=False, bias=True)
        self.conv4 = torch.nn.Linear(num_hidden, in_dim, bias=True)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1)
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = self.conv4(h3)
        return h2, h4


class CrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, alpha=0.2):
        super(CrossAttention, self).__init__()

        self.dropout = dropout
        self.NFeature = dim
        self.alpha = alpha

        self.a1 = nn.Parameter(torch.FloatTensor(2 * dim, 1))
        self.a2 = nn.Parameter(torch.FloatTensor(2 * dim, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def inference(self, h1=None, h2=None):
        e = self._prepare_attentional_mechanism_input(h1, h2)
        lamda = F.softmax(e, dim=1)
        lamda = F.dropout(lamda, self.dropout, training=self.training)

        h_prime1 = lamda[:, 0].repeat(self.NFeature, 1).T * h1
        h_prime2 = lamda[:, 1].repeat(self.NFeature, 1).T * h2

        h = h_prime1 + h_prime2

        return lamda, h

    def forward(self, h1=None, h2=None):
        lamda, h = self.inference(h1, h2)

        return lamda, h

    def _prepare_attentional_mechanism_input(self, h1=None, h2=None):
        Wh1 = torch.matmul(torch.cat([h1, h2], 1), self.a1)
        Wh2 = torch.matmul(torch.cat([h1, h2], 1), self.a2)
        # broadcast add
        e = torch.cat((Wh1, Wh2), 1)

        return self.leakyrelu(e)


class SPOT(Module):
    def __init__(self, dim, dims_spot=[64, 16], dropout=0.1, device='cuda:0', cluster=True):
        super(SPOT, self).__init__()

        self.dropout = dropout
        self.NFeature = dim
        self.device = device
        self.cluster = cluster
        if cluster:
            self.GATE_spot = GAT(hidden_dims=dims_spot).to(device)
        else:
            self.GATE_spot = GAT_noncluster(hidden_dims=dims_spot).to(device)
        self.CrossAtt = CrossAttention(dim=self.NFeature, dropout=0.0)

    def forward(self, gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T, spotnet, spotnet_cluster):
        if self.cluster:
            emb_spot, mtx_res_spot = self.GATE_spot(spot_freq_mtx, spotnet, spotnet_cluster)
        else:
            emb_spot, mtx_res_spot = self.GATE_spot(spot_freq_mtx, spotnet)
        X_res_gene = self.iGFT_gene(gene_freq_mtx, gene_eigvecs_T)
        X_res_spot = self.iGFT_spot(mtx_res_spot, spot_eigvecs_T)
        lamba, res = self.CrossAtt(X_res_gene, X_res_spot)

        return res, lamba, emb_spot, mtx_res_spot

    def iGFT_gene(self, mtx_res, eigvecs_T):
        filter = torch.matmul(eigvecs_T.T, mtx_res.T)
        filter[filter < 0] = 0
        return filter

    def iGFT_spot(self, mtx_res, eigvecs_T):
        filter = torch.matmul(mtx_res, eigvecs_T)
        filter[filter < 0] = 0
        return filter


class GENE(Module):
    def __init__(self, dims_gene=[64, 16], dropout=0.02, device='cuda:0'):
        super(GENE, self).__init__()
        self.dropout = dropout
        self.device = device

        self.gat1 = GraphAttentionLayer(dims_gene[0], dims_gene[1], dropout=dropout, alpha=0.2, concat=True)
        self.gat2 = GraphAttentionLayer(dims_gene[1], dims_gene[2], dropout=dropout, alpha=0.2, concat=False)
        self.AdjLayer = InnerProductDecoder(dropout=dropout)

    def forward(self, features, adj):
        h1 = self.gat1(features, adj, tied_W=None, tied_attention=None)
        attention = self.gat1.attention_res
        h2 = self.gat2(h1, adj, tied_W=None, tied_attention=None)

        genenet_res = self.AdjLayer(h2) * (adj.sum(axis=1) > 0).reshape(-1, 1)

        return h2, genenet_res, attention

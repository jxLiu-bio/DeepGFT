import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, tied_W=None, tied_attention=None):
        self.tied_W = tied_W
        self.tied_attention = tied_attention
        if self.tied_W is not None:
            # self.W = self.tied_W
            Wh = torch.mm(h, self.tied_W)
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        self.W_res = self.W
        e = self._prepare_attentional_mechanism_input(Wh)

        if self.tied_attention is not None:
            attention = self.tied_attention
        else:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            # attention = F.dropout(attention, self.dropout, training=self.training)

        self.attention_res = attention
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class InnerProductDecoder(Module):
    def __init__(self, dropout, act=F.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj




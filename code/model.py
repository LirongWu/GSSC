import torch
import torch.nn as nn

import pyro


def FC(in_dim, out_dim, dropout):
    return nn.Sequential(nn.BatchNorm1d(in_dim), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))


# Architecture: an MLP and two prediction heads
class MLP_GNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, param):
        super(self.__class__, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(param['n_layers'] - 1):
            self.layers.append(FC(hid_dim, hid_dim, param['dropout']))

        self.out = FC(hid_dim, out_dim, param['dropout'])
        self.inf = FC(hid_dim, out_dim, param['dropout'])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.out(x), self.inf(x)

    def update(self, x):
        for layer in self.layers:
            x = layer(x)

        y_pred = torch.argmax(self.out(x), dim=1)

        return y_pred

    def com_hom_loss(self, y_pred, src, dst, weight):

        neg_samples = weight[y_pred[src] != y_pred[dst]]
        pos_samples = weight[y_pred[src] == y_pred[dst]]

        loss_hom_mean = 1.0 * torch.mean(neg_samples) / torch.mean(pos_samples + 1e-12)
        loss_hom_sum = 1.0 * torch.sum(neg_samples) / torch.sum(pos_samples + 1e-12)

        return loss_hom_mean, loss_hom_sum


# Neighborhood Sparsification (NeighSparse) Network
class EdgeSampler(nn.Module):
    def __init__(self, in_dim, hid_dim, param):
        super(EdgeSampler, self).__init__()

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        
        self.alpha = param['alpha']
        self.temp = param['temp']

    def forward(self, x, src, dst, adj_weight):

        h = self.linear1(x)
        h = self.linear2(h)

        # Sparsification Distribution
        edge_logits = torch.sigmoid(torch.sum(h[src] * h[dst], dim=1))
        edge_probs = edge_logits * (1-self.alpha) + self.alpha * adj_weight

        # Subgraph Sampling
        edge_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temp, probs=edge_probs).rsample()

        return edge_sampled


# Positive sample augmentation by virtual node generation
class MixupScale(nn.Module):
    def __init__(self, in_dim, hid_dim, param):
        super(MixupScale, self).__init__()

        self.linear = nn.Linear(in_dim, hid_dim)
        self.att = nn.Linear(hid_dim*2, 1, bias=False)
        self.norm = nn.BatchNorm1d(hid_dim)
        self.scale = param['scale']

    def forward(self, x, src, dst):
        x_src = self.norm(self.linear(x[src]))
        x_dst = self.norm(self.linear(x[dst]))
        h = torch.cat((x_src, x_dst), dim=1)
        ratio = torch.sigmoid(self.att(h))

        return ratio * (1-self.scale) + self.scale
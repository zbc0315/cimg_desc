#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:53
# @Author  : zhangbc0315@outlook.com
# @File    : model_retro_rcts.py
# @Software: PyCharm

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
import torch_scatter
from torch.nn import Module, Sequential, Linear, ELU


class EdgeModel(Module):
    def __init__(self, num_node_features, num_edge_features, out_features):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Sequential(Linear(num_node_features + num_node_features + num_edge_features, 128),
                                   ELU(),
                                   # Dropout(0.5),
                                   Linear(128, out_features))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(Module):
    def __init__(self, num_node_features, num_edge_features_out, out_features):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Sequential(Linear(num_node_features + num_edge_features_out, 256),
                                     ELU(),
                                     # Dropout(0.5),
                                     Linear(256, 256))
        self.node_mlp_2 = Sequential(Linear(num_node_features + 256, 256),
                                     ELU(),
                                     # Dropout(0.5),
                                     Linear(256, out_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = torch_scatter.scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(Module):
    def __init__(self, num_node_features, num_global_features, out_channels):
        super(GlobalModel, self).__init__()
        self.global_mlp = Sequential(Linear(num_global_features + num_node_features, 256),
                                     ELU(),
                                     # Dropout(0.3),
                                     Linear(256, out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        if u is None:
            out = torch_scatter.scatter_mean(x, batch, dim=0)
        else:
            out = torch.cat([u, torch_scatter.scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        num_node_features = 10
        num_edge_features = 5
        out_channels = 14541
        self.node_normal = gnn.BatchNorm(num_node_features)
        self.edge_normal = gnn.BatchNorm(num_edge_features)
        # self.node_layer = nn.Linear(num_node_features, num_node_features)
        # self.edge_layer = nn.Linear(num_edge_features, num_edge_features)
        self.meta1 = gnn.MetaLayer(EdgeModel(num_node_features, num_edge_features, 512),
                                   NodeModel(num_node_features, 512, 128),
                                   GlobalModel(128, 0, 128))
        self.meta2 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta3 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta4 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta5 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta6 = gnn.MetaLayer(EdgeModel(128, 512, 128),
                                   NodeModel(128, 128, 128),
                                   GlobalModel(128, 128, 256))
        self.lin1 = nn.Linear(256, 512)
        self.lin2 = nn.Linear(512, out_channels)

    def forward(self, data):
        # x, edge_index, e, g, batch = data.x, data.edge_index, data.edge_attr, data.g, data.batch
        x, edge_index, e, g, batch = data.x, data.edge_index, data.edge_attr, None, data.batch
        # x = x[:, :-2]
        x = self.node_normal(x)
        e = self.edge_normal(e)

        x, e, g = self.meta1(x, edge_index, e, g, batch)
        # x = F.dropout(x, 0.3)
        x, e, g = self.meta2(x, edge_index, e, g, batch)
        # x = F.dropout(x, 0.3)
        x, e, g = self.meta3(x, edge_index, e, g, batch)
        # x = F.dropout(x, 0.3)
        x, e, g = self.meta4(x, edge_index, e, g, batch)
        # x = F.dropout(x, 0.3)
        x, e, g = self.meta5(x, edge_index, e, g, batch)
        # x = F.dropout(x, 0.3)
        x, e, g = self.meta6(x, edge_index, e, g, batch)

        y = F.elu(self.lin1(g))
        return g, self.lin2(y)


if __name__ == "__main__":
    pass

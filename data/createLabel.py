import sys
sys.path.append("..") #相对路径或绝对路径

import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from utils.utils_algo import *
import os.path as osp
from datasets.dataset import get_dataset
from utils.utils_dataset import generate_split, graph_aug
from torch_geometric.nn import ARMAConv, global_mean_pool
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch.nn import ModuleList

torch.set_printoptions(precision=2, sci_mode=False)


class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=128, feat_dim=128, num_class=2, num_layers=1):
        super(ARMA, self).__init__()

        self.convs = ModuleList([
            ARMAConv(in_channels, hid_channels),
            ARMAConv(hid_channels, hid_channels, num_layers=num_layers)])
        self.causal_mlp = torch.nn.Sequential(
            Linear(hid_channels, 2*hid_channels),
            ReLU(),
            Linear(2*hid_channels, num_class)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)

        logits = self.causal_mlp(graph_x)
        return logits

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        temp = self.convs[0](x, edge_index, edge_weight)
        x = F.relu(temp)
        node_x = self.convs[1](x, edge_index, edge_weight)
        return node_x

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

    def save(self, path):
        torch.save(self.state_dict(), path)


def createNoise(dataset, in_channels, partial_rate, modelPath,topk=5):
    label_lis = [g.y for g in dataset]
    labels = torch.cat(label_lis, dim=0)

    max_label = torch.max(labels).item()
    min_label = torch.min(labels).item()
    num_class = max_label - min_label + 1
    model = ARMA(in_channels=in_channels,num_class=num_class)

    # load model
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])

    outputs = []
    for graph in dataset:
        output = model(graph.x, graph.edge_index)
        outputs.append[output]
    outputs = torch.cat(outputs, dim=0)

    # ground-truth
    y=labels.unsqueeze(1)
    outputs.scatter_(dim=1, index=y, value=0)

    topk_values, topk_indices = torch.topk(outputs, topk, dim=1)

    partialY = torch.zeros(outputs.shape[0], outputs.shape[1])

    mask = torch.rand_like(outputs) < partial_rate
    mask = mask.to(partialY.dtype)
    partialY.scatter_(1, topk_indices, mask)

    partialY.scatter_(dim=1, index=y, value=1)

    return partialY
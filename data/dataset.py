import networkx as nx
import numpy as np
from torch.utils import data
import torch


def Read_graph(file_name):
    edge = np.loadtxt(file_name).astype(np.int32)
    min_node, max_node = edge.min(), edge.max()
    if min_node == 0:
        Node = max_node + 1
    else:
        Node = max_node
    G = nx.Graph()
    Adj = np.zeros([Node, Node], dtype=np.int32)
    for i in range(edge.shape[0]):
        G.add_edge(edge[i][0], edge[i][1])
        if min_node == 0:
            Adj[edge[i][0], edge[i][1]] = 1
            Adj[edge[i][1], edge[i][0]] = 1
        else:
            Adj[edge[i][0] - 1, edge[i][1] - 1] = 1
            Adj[edge[i][1] - 1, edge[i][0] - 1] = 1
    for i in range(Node):
        Adj[i][i] = 1
    Adj = torch.FloatTensor(Adj)
    return G, Adj, Node


class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.Node
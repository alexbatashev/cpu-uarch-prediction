import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.functional import F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gcn1 = GCNConv(input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

    def forward(self, basic_block, edge_index):
        x = F.relu(self.gcn1(basic_block, edge_index))
        x = self.gcn2(x, edge_index)

        return x

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)

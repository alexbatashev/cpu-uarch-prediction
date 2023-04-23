import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gcn = GCNConv(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, basic_block, edge_index, hidden):
        edge_index, _ = add_self_loops(edge_index, num_nodes=basic_block.size(0))
        x = self.gcn(basic_block, edge_index)

        output, hidden = self.gru(x, hidden)
        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gcn = GCNConv(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, basic_block, edge_index, hidden):
        x = self.gcn(basic_block, edge_index)
        x = x.unsqueeze(1)
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)

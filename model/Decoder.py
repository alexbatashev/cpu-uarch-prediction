import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.functional import F


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, output_dim)
        #self.fc2 = nn.Linear(output_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.relu(self.gat1(x, edge_index))
        # x = self.fc2(x)
        return x

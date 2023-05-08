import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, PositionalEncoding
from torch.functional import F


class GCNEncoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_opcodes):
        super(GCNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gcn1 = GCNConv(num_opcodes, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

    def forward(self, basic_block, edge_index):
        x = self.gcn1(basic_block, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)

        return x


class GATEncoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_heads=2, dtype=torch.float32):
        super(GATEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, emb_size, dtype=dtype)
        self.gat1 = GATConv(emb_size, hidden_size, heads=num_heads, concat=True, dtype=dtype)
        self.gat2 = GCNConv(hidden_size * num_heads, hidden_size, heads=1, concat=False, dtype=dtype)

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        return x

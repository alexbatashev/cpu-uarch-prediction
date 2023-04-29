import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, PositionalEncoding
from torch.functional import F


class OpcodeEmbedding(nn.Module):
    def __init__(self, num_opcodes, embedding_dim):
        super(OpcodeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_opcodes, embedding_dim)
        self.num_features = 5  # the number of features without opcode

    def forward(self, x):
        opcode_embedding = self.embedding(x[:, -1].long())
        other_features = x[:, :self.num_features]
        return torch.cat((other_features, opcode_embedding), dim=1)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_opcodes):
        super(Encoder, self).__init__()
        self.embedding = OpcodeEmbedding(num_opcodes, embedding_dim)
        self.hidden_size = hidden_size
        self.gcn1 = GCNConv(input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size)

    def forward(self, basic_block, edge_index):
        #basic_block = self.pos_enc(basic_block)
        x = self.gcn1(basic_block, edge_index)
        x = self.pos_enc(x)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)

        return x

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)

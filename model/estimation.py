from torch.nn import Module, Linear, LSTM, Dropout2d, BatchNorm1d, GRU
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, GraphNorm, BatchNorm
from torch_geometric.utils import to_dense_batch
import torch


class GNNEstimation(Module):
    def __init__(self, conv, input_dim, hidden_dim, output_dim, batch_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = torch.nn.Embedding(21000, input_dim)
        self.norm1 = GraphNorm(input_dim)

        if conv == 'GraphConv':
            self.conv = GraphConv(input_dim, hidden_dim)
        elif conv == "GCNConv":
            self.conv = GCNConv(input_dim, hidden_dim)
        else:
            raise "Failed"

        self.norm2 = BatchNorm(hidden_dim)

        self.bidirectional = True

        self.lstm = LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional)

        self.concat = Linear(hidden_dim * 2, hidden_dim)
        self.dropout = Dropout2d(p=0.3)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.embedding(x)

        x = self.conv(x, edge_index)
        # x = F.relu(x)

        x = self.norm2(x)

        nodes, mask = to_dense_batch(x, batch)

        x, _ = self.lstm(nodes[:, 1:, :])
        x = x[:, -1, :]
        x = x.reshape(self.batch_size, -1)
        if self.bidirectional:
            x = self.concat(x)
            x = F.relu(x)
        x = self.fc(x)
        x = F.relu(x)
        return x.squeeze()


class LSTMEstimation(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = torch.nn.Embedding(21000, input_dim, scale_grad_by_freq=True)

        self.norm1 = BatchNorm1d(input_dim)

        self.bidirectional = True

        self.lstm = LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional)

        self.concat = Linear(2 * hidden_dim, hidden_dim)

        self.norm2 = BatchNorm1d(hidden_dim)

        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        batch = data.batch

        x = self.embedding(x)

        nodes, mask = to_dense_batch(x, batch)

        # FIXME this is only allowed when we have has_virtual_root=True
        nodes = nodes[:, 1:, :]

        #nodes = torch.transpose(nodes, 1, 2)
        #nodes = self.norm1(nodes)
        #nodes = torch.transpose(nodes, 1, 2)

        x, _ = self.lstm(nodes)
        x = x[:, 0, :]
        x = x.reshape(self.batch_size, -1)

        if self.bidirectional:
            x = self.concat(x)
            x = F.relu(x)

        x = self.fc(x)
        x = F.relu(x)
        return x.squeeze()

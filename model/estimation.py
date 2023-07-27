from torch.nn import Module, Linear, LSTM
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv, GraphNorm, BatchNorm
from torch_geometric.utils import to_dense_batch
import torch


class GNNEstimation(Module):
    def __init__(self, conv, input_dim, hidden_dim, output_dim, batch_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # self.embedding = TemporalEncoding(hidden_dim)
        self.embedding = torch.nn.Embedding(21000, input_dim)
        self.norm1 = GraphNorm(input_dim)

        if conv == 'GraphConv':
            self.conv = GraphConv(input_dim, hidden_dim)
        elif conv == "GCNConv":
            self.conv = GCNConv(input_dim, hidden_dim)
        else:
            raise "Failed"

        self.norm2 = BatchNorm(hidden_dim)

        # self.fc1 = Linear(hidden_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc2 = Linear(hidden_dim, output_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros([1, self.batch_size, self.hidden_dim]).cuda().detach(), torch.zeros([1, self.batch_size, self.hidden_dim]).cuda().detach()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # x = self.norm1(x)

        x = self.embedding(x)

        x = self.conv(x, edge_index)
        x = F.relu(x)

        x = self.norm2(x)

        nodes, mask = to_dense_batch(x, batch)

        # nodes = self.fc1(nodes)
        # nodes = F.relu(nodes)

        x, hidden = self.lstm(nodes[:, 1:, :], self.hidden)
        h0, h1 = hidden
        self.hidden = (h0.detach(), h1.detach())
        x = x[:, -1, :]
        x = x.reshape(self.batch_size, -1)
        x = self.fc2(x)
        x = F.relu(x)
        return x.squeeze()


class LSTMEstimation(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embedding = torch.nn.Embedding(21000, input_dim)

        # self.norm = BatchNorm(input_dim)

        self.lstm = LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = Linear(hidden_dim, output_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.rand([1, self.batch_size, self.hidden_dim]).cuda().detach(), torch.rand([1, self.batch_size, self.hidden_dim]).cuda().detach()

    def forward(self, data):
        x = data.x
        batch = data.batch

        # x = self.norm(x)

        x = self.embedding(x)

        nodes, mask = to_dense_batch(x, batch)

        x, hidden = self.lstm(nodes, self.hidden)
        h0, h1 = hidden
        self.hidden = (h0.detach(), h1.detach())
        x = x[:, -1, :]
        x = x.reshape(self.batch_size, -1)
        x = F.relu(x)
        x = self.fc(x)
        x = F.relu(x)
        return x.squeeze()

import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, encoder, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

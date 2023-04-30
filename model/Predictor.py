import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, encoder, hidden_size, output_size):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_size, output_size)
        self.threshold = nn.Threshold(0.2, 0)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.fc(x)
        x = self.relu(x)
        #x = self.threshold(x)
        return x

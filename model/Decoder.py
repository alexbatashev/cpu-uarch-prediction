import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        output = self.relu(output)
        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size, device=device)

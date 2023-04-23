import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, input_size, nhead, hidden_size):
        super(Transformer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size)

    def forward(self, x):
        x = self.transformer(x)
        return x
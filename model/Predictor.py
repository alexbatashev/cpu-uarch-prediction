import torch.nn as nn
from torch_geometric.utils import add_self_loops


class Predictor(nn.Module):
    def __init__(self, encoder, transformer, decoder):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.transformer = transformer
        self.decoder = decoder

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        encoded_x = self.encoder(x, edge_index).unsqueeze(1)
        transformed_x = self.transformer(encoded_x).squeeze(1)
        port_pressures = self.decoder(transformed_x, edge_index)
        return port_pressures, transformed_x

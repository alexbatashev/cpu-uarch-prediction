import torch
import torch.nn as nn


class RlAgent(nn.Module):
    def __init__(self, encoder, decoder):
        super(RlAgent, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_sequence, edge_index, hidden):
        encoder_output, encoder_hidden = self.encoder(input_sequence, edge_index, hidden)
        decoder_input = torch.tensor([[0.0]])  # Placeholder input for decoder
        decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden)
        return decoder_output

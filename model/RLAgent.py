import torch
import torch.nn as nn


class RlAgent(nn.Module):
    def __init__(self, encoder, decoder):
        super(RlAgent, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # FIXME need proper device
        self.encoder_hidden = encoder.init_hidden(torch.device("cuda"))
        self.decoder_hidden = decoder.init_hidden(torch.device("cuda"))

    def forward(self, input_sequence, edge_index):
        encoder_output, encoder_hidden = self.encoder(input_sequence, edge_index, self.encoder_hidden)
        self.encoder_hidden = encoder_hidden.detach()
        decoder_output, decoder_hidden = self.decoder(encoder_output, self.decoder_hidden)
        self.decoder_hidden = decoder_hidden.detach()
        return decoder_output

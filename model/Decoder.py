import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, input, hidden):
        #print(input.size(-1))
        #input = input.view(-1, 1)
        output, hidden = self.lstm(input, hidden)
        return output, hidden

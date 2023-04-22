import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gcn = GCNConv(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, basic_block, edge_index, hidden):
        edge_index, _ = add_self_loops(edge_index, num_nodes=basic_block.size(0))
        x = self.gcn(basic_block, edge_index)
        max_seq_len = 256
        #x = x.unsqueeze(1)
        #padded_x = torch.zeros(max_seq_len, 1, self.hidden_size).to(x.device)
        #padded_x[:x.size(0), :, :] = x
        #x = padded_x
        # x = x.expand(-1, max_seq_len, -1)
        #padded_x = torch.zeros((max_seq_len, 1, self.hidden_size), device=x.device)
        #padded_x[:x.size(0), :, :] = x
        #x = padded_x
        #hidden = (hidden[0][:,:1,:], hidden[1][:,:1,:])
        #hidden = hidden[0][:,:1,:]

        #print(hidden[0].shape())
        #print(x.shape())
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, device):
        max_seq_len = 256
        return torch.zeros(1, self.hidden_size).to(device), torch.zeros(1, self.hidden_size).to(device)

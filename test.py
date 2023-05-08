import os

embedding_size = 128
hidden_size = 64
batch_size = 1
output_size = 10 # It is known that Willow Cove features 10 "ports"
num_heads = 2

learning_rate = 0.001

import model.utils
import torch
import torch.utils.data
from torch_geometric.loader import DataLoader

dtype = torch.float32

dataset = model.utils.BasicBlockDataset("data/i5_1135g7.pb", dtype=dtype)
train_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

from model.GraphEncoder import GATEncoder
from model.Predictor import Predictor
import torch_geometric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

encoder = GATEncoder(dataset.num_opcodes, embedding_size, hidden_size, num_heads, dtype=dtype).to(device)
model = Predictor(encoder, hidden_size, output_size, dtype=dtype).to(device)
#model = torch_geometric.compile(model)

from model.model import train

checkpoint_dir = "checkpoints/tgl"
checkpoint_freq = 10
num_epochs = 200

if torch.cuda.is_available():
    torch.cuda.empty_cache()

train(model, device, loader, num_epochs, batch_size, learning_rate, checkpoint_dir, checkpoint_freq)

torch.save(model, "trained_models/tgl.pt")

choice = dataset[140]
bb, m, raw = choice

print(bb.x)

input_sequence = bb.x.to(device)
edge_index = bb.edge_index.to(device)

out, _ = model(input_sequence, edge_index)
res = out.to("cpu").detach().numpy()
model.utils.print_port_pressure_table(res[1:], raw["source"])
print(model.utils.estimate_cycles(out))
print(m)
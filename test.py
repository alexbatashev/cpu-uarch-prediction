import model.utils
import torch
import sys

inputs = model.utils.load_all_basic_blocks_data("data/x86_64/basic_blocks")

measured_cycles = model.utils.load_measured_data("data/x86_64/ryzen3600")

input_size = 6 # By the number of features extracted with llvm-mc-embed
hidden_size = 128
output_size = 10
nhead = 2
learning_rate = 0.000001

from model.GraphEncoder import Encoder
from model.Decoder import Decoder
from model.Transformer import Transformer
from model.Predictor import Predictor

if sys.platform == "darwin":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(input_size, hidden_size, output_size).to(device)
decoder = Decoder(input_size, hidden_size, output_size).to(device)
transformer = Transformer(input_size, nhead, hidden_size).to(device)
agent = Predictor(encoder, transformer, decoder).to(device)

from model.model import train
import random

checkpoint_dir = "checkpoints/ryzen3600"

checkpoint_freq = 100

num_epochs = 50

data = [(x, y) for x, y in zip(inputs, measured_cycles)]

train(agent, device, data, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq)

choice = random.choice(data)
bb, m = choice

input_sequence = bb.x.to("cuda")
edge_index = bb.edge_index.to("cuda")

out, _ = agent(input_sequence, edge_index)
res = out.to("cpu").detach().numpy()
model.utils.print_port_pressure_table(res)
print(model.utils.estimate_cycles(out))
print(m)

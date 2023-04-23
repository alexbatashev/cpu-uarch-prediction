import model.utils

inputs = model.utils.load_all_basic_blocks_data("data/x86_64/basic_blocks")

measured_cycles = model.utils.load_measured_data("data/x86_64/ryzen3600")

input_size = 6 # By the number of features extracted with llvm-mc-embed
hidden_size = 128
output_size = 10
learning_rate = 0.001

from model.GraphEncoder import Encoder
from model.Decoder import Decoder
from model.RLAgent import RlAgent

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
agent = RlAgent(encoder, decoder)

from model.model import train
import random

checkpoint_dir = "checkpoints/ryzen3600"

checkpoint_freq = 500

num_epochs = 5000

data = [(x, y) for x, y in zip(inputs, measured_cycles)]

train(encoder, decoder, agent, data, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq)

choice = random.choice(data)
bb, m = choice

input_sequence = bb.x.to("cuda")
edge_index = bb.edge_index.to("cuda")

out = agent(input_sequence, edge_index)
res = out.to("cpu").detach().numpy()
print(res)
print(model.utils.estimate_cycles(out))
print(m)

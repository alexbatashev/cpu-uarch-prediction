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

checkpoint_freq = 5

num_epochs = 100

data = [(x, y) for x, y in zip(inputs, measured_cycles)]

choice = random.choice(data)
bb, m = choice
print(bb.x)
print(bb.x.size(0))

train(encoder, decoder, agent, data, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq)

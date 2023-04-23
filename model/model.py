import random
import torch
import os
from model.reward import reward_function
import torch.optim as optim
import model.utils as utils
from tqdm import tqdm
from sys import platform
from torch.utils.tensorboard import SummaryWriter


def train(encoder, decoder, agent, inputs, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq=50):
    # Check if a GPU is available and set the device accordingly
    if platform == "darwin":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()

    # Move the models to the device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    agent = agent.to(device)

    optimizer = optim.SGD(agent.parameters(), lr=learning_rate)

    # Load the most recent checkpoint if it exists
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        #checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)
        checkpoint_files = os.listdir(checkpoint_dir)
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])

    if latest_checkpoint:
        start_epoch = utils.load_checkpoint(latest_checkpoint, agent, optimizer)
    else:
        start_epoch = 0

    for i in tqdm(range(start_epoch, num_epochs)):
        choice = random.choice(inputs)
        bb, measured = choice
        input_sequence = bb.x.to(device)
        edge_index = bb.edge_index.to(device)

        port_pressures = agent(input_sequence, edge_index)
        predicted_cycles = utils.estimate_cycles(port_pressures)


        reward = reward_function(port_pressures, predicted_cycles, measured, bb.x)

        loss = torch.sum(port_pressures * reward)
        writer.add_scalar("Loss/train", loss)
        loss.backward()

        if (i + 1) % checkpoint_freq == 0:
            utils.save_checkpoint(i + 1, agent, optimizer, checkpoint_dir)

    writer.flush()
    #writer.add_graph(agent)
    #writer.flush()
    writer.close()

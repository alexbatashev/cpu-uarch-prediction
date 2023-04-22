import random
import torch
import os
from reward import reward_function
import torch.nn as nn
import torch.optim as optim
import utils
from tqdm import tqdm


def train(encoder, decoder, agent, inputs, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq=50):
    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Move the models to the device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    agent = agent.to(device)

    # Load the most recent checkpoint if it exists
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])

    if latest_checkpoint:
        start_epoch = utils.load_checkpoint(latest_checkpoint, agent, optimizer)
    else:
        start_epoch = 0

    for i in tqdm(range(start_epoch, num_epochs)):
        bb, measured = random.choice(inputs)
        input_sequence = bb.x.to(device)
        edge_index = bb.edge_index.to(device)

        hidden = encoder.init_hidden().to(device)

        port_pressures = agent(input_sequence, edge_index, hidden)
        predicted_cycles = utils.estimate_cycles(port_pressures)

        reward = reward_function(port_pressures, predicted_cycles, measured, bb.x)

        loss = -torch.sum(port_pressures * reward)
        loss.backward()
        optimizer.step()

        if (i + 1) % checkpoint_freq == 0:
            utils.save_checkpoint(i + 1, agent, optimizer, checkpoint_dir)

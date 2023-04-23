import random
import torch
import os
from model.reward import reward_function
import torch.optim as optim
import model.utils as utils
from tqdm import tqdm
from sys import platform
from torch.utils.tensorboard import SummaryWriter
from torch.functional import F


def loss_function(predicted_port_pressures, measured_cycles, alpha=0.1):

    # Regularization term (L1 sparsity constraint)
    regularization_term = torch.sum(torch.abs(predicted_port_pressures))

    # Measured cycles term
    total_predicted_cycles = torch.sum(torch.sum(predicted_port_pressures, dim=1))
    measured_cycles_term = F.mse_loss(total_predicted_cycles, torch.tensor([measured_cycles], device=total_predicted_cycles.device))

    # Combine the losses
    total_loss = alpha * regularization_term + measured_cycles_term

    return total_loss


def train(predictor, device, inputs, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq=50):
    # Check if a GPU is available and set the device accordingly

    writer = SummaryWriter()

    optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

    predictor.train()
    losses = []

    # Load the most recent checkpoint if it exists
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        #checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)
        checkpoint_files = os.listdir(checkpoint_dir)
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])

    if latest_checkpoint:
        start_epoch = utils.load_checkpoint(latest_checkpoint, predictor, optimizer)
    else:
        start_epoch = 0

    for i in tqdm(range(start_epoch, num_epochs)):
        optimizer.zero_grad()
        choice = random.choice(inputs)
        bb, measured = choice
        input_sequence = bb.x.to(device)
        edge_index = bb.edge_index.to(device)

        port_pressures, tranformed_x = predictor(input_sequence, edge_index)
        predicted_cycles = utils.estimate_cycles(port_pressures)

        loss = loss_function(port_pressures, measured)
        losses.append(loss)

        writer.add_scalar("Loss/train", loss, i)
        loss.backward()

        if (i + 1) % checkpoint_freq == 0:
            utils.save_checkpoint(i + 1, predictor, optimizer, checkpoint_dir)

    writer.flush()
    choice = random.choice(inputs)
    bb, measured = choice
    input_sequence = bb.x.to(device)
    edge_index = bb.edge_index.to(device)
    writer.add_graph(predictor, [input_sequence, edge_index])
    #writer.flush()
    writer.close()
    print(losses)

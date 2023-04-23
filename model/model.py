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


def loss_function(predicted_port_pressures, measured_cycles, nodes, alpha=0.1):

    # Regularization term (L1 sparsity constraint)
    regularization_term = torch.sum(torch.abs(predicted_port_pressures))

    # Measured cycles term
    max_pp = torch.max(predicted_port_pressures, dim=1)
    total_predicted_cycles = torch.sum(max_pp.values)
    measured_cycles_term = F.mse_loss(total_predicted_cycles, torch.tensor(measured_cycles, device=total_predicted_cycles.device))

    # TODO proper mapping
    # 2 -> is_load
    # 3 -> is_store
    # 1 -> is_compute

    base_scale_factor = 1.0

    pure_memory_instructions = [
        i for i, node in enumerate(nodes) if (node[2] or node[3]) and not node[1]
    ]

    pure_compute_instructions = [
        i for i, node in enumerate(nodes) if not (node[2] or node[3]) and node[1]
    ]

    port_pressures = predicted_port_pressures.detach().numpy()
    for i in pure_memory_instructions:
        for j in pure_compute_instructions:
            for k in range(0, len(port_pressures[0])):
                if port_pressures[i][k] != 0 and port_pressures[j][k] != 0:
                    base_scale_factor += 0.2

    # Combine the losses
    total_loss = alpha * regularization_term + measured_cycles_term

    return base_scale_factor * total_loss


def train(predictor, device, loader, num_epochs, learning_rate, checkpoint_dir, checkpoint_freq=50):
    # Check if a GPU is available and set the device accordingly

    writer = SummaryWriter()

    optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

    predictor.train()

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

    epochs = num_epochs - start_epoch
    pbar = tqdm(range(epochs * len(loader)))
    optimizer.zero_grad()
    for i in range(start_epoch, num_epochs):
        for choice in loader:
            #choice = loader.next()
            bb, measured, _ = choice
            input_sequence = bb.x.to(device)
            edge_index = bb.edge_index.to(device)

            port_pressures, _ = predictor(input_sequence, edge_index)

            loss = loss_function(port_pressures, measured, bb.x)

            writer.add_scalar("Loss/train", loss, i)
            loss.backward()

            if (i + 1) % checkpoint_freq == 0:
                utils.save_checkpoint(i + 1, predictor, optimizer, checkpoint_dir)
            pbar.update()

    writer.flush()
    choice = random.choice(loader)
    bb, measured, _ = choice
    input_sequence = bb.x.to(device)
    edge_index = bb.edge_index.to(device)
    writer.add_graph(predictor, [input_sequence, edge_index])
    writer.close()

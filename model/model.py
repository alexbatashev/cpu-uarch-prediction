import random

import numpy
import torch
import os

import model.reward
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
    measured_cycles_term = F.mse_loss(total_predicted_cycles,
                                      torch.tensor(measured_cycles, device=total_predicted_cycles.device))

    # TODO proper mapping
    # 2 -> is_load
    # 3 -> is_store
    # 1 -> is_compute

    base_scale_factor = 1.0

    # print(nodes)
    #
    # port_pressures = predicted_port_pressures.detach().cpu().numpy()
    #
    # rewards = []
    #
    # print(len(port_pressures))
    # print(len(port_pressures[0]))
    # for i, p in enumerate(port_pressures):
    #     print(i)
    #     rewards.append(model.reward.reward_lsu_compute_separation(p, nodes, i - 1))
    #
    # base_scale_factor += numpy.average(rewards)

    total_loss = alpha * regularization_term + measured_cycles_term

    return total_loss * base_scale_factor


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
    step = 0
    for i in range(start_epoch, num_epochs):
        train_loss = 0
        for choice in loader:
            optimizer.zero_grad()
            #choice = loader.next()
            bb, measured, raw = choice
            input_sequence = bb.x.to(device)
            edge_index = bb.edge_index.to(device)

            port_pressures, _ = predictor(input_sequence, edge_index)

            loss = loss_function(port_pressures, measured, raw["nodes"])
            train_loss += loss.item()

            step += 1
            loss.backward()
            optimizer.step()

            if (i + 1) % checkpoint_freq == 0:
                utils.save_checkpoint(i + 1, predictor, optimizer, checkpoint_dir)
            pbar.update()
        writer.add_scalar("Loss/train", train_loss, i)

    writer.flush()
    choice = random.choice(loader.dataset)
    bb, measured, _ = choice
    input_sequence = bb.x.to(device)
    edge_index = bb.edge_index.to(device)
    writer.add_graph(predictor, [input_sequence, edge_index])
    writer.close()

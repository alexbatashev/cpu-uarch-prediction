import random
import torch
import os
import numpy as np
import torch.optim as optim
import model.utils as utils
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn


def loss_function(predicted_port_pressures, measured_cycles, batch, nodes, alpha=0.1):
    cpu_batch = batch.detach().to(torch.device("cpu")).numpy()
    split_predictions = torch.zeros(measured_cycles.shape[0])

    for i in range(0, measured_cycles.shape[0]):
        all_max = []
        for idx, b in enumerate(cpu_batch):
            if b == i:
                all_max.append(torch.max(predicted_port_pressures[idx]))
        if len(all_max) > 0:
            all_max = torch.stack(all_max)
            split_predictions[i] = torch.sum(all_max)

    criterion = nn.MSELoss()

    return criterion(split_predictions, measured_cycles)


def train(predictor, device, loader, num_epochs, batch_size, learning_rate, checkpoint_dir, checkpoint_freq=50):
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
        all_losses = []
        for choice in loader:
            optimizer.zero_grad()
            bb, measured, raw = choice
            input_sequence = bb.x.to(device)
            edge_index = bb.edge_index.to(device)

            port_pressures = predictor(input_sequence, edge_index)

            loss = loss_function(port_pressures, measured, bb.batch, [])
            all_losses.append(loss.item())

            step += 1
            loss.backward()
            optimizer.step()

            if (i + 1) % checkpoint_freq == 0:
                utils.save_checkpoint(i + 1, predictor, optimizer, checkpoint_dir)
            pbar.update()
        writer.add_scalar("Total Loss/train", np.sum(all_losses), i)
        writer.add_scalar("Avg Loss/train", np.average(all_losses), i)

    writer.flush()
    choice = random.choice(loader.dataset)
    bb, measured, _ = choice
    input_sequence = bb.x.to(device)
    edge_index = bb.edge_index.to(device)
    writer.add_graph(predictor, [input_sequence, edge_index])
    writer.close()

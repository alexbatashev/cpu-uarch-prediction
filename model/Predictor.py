import torch.nn as nn
import lightning.pytorch as pl
import torch
import deepspeed


def loss_function(predicted_port_pressures, measured_cycles, batch, has_virt_node):
    device = predicted_port_pressures.device
    dtype = predicted_port_pressures.dtype

    batch_size = measured_cycles.shape[0]

    cpu_batch = batch.detach().to(torch.device("cpu")).numpy()
    split_predictions = [[]] * batch_size

    for idx, b in enumerate(cpu_batch):
        split_predictions[b].append(predicted_port_pressures[idx])

    predicted_cycles = torch.zeros(batch_size, device=device, dtype=dtype)

    for idx, s in enumerate(split_predictions):
        tensor = torch.stack(s).to(device)
        if has_virt_node:
            tensor = tensor[1:]

        sum = tensor.sum(dim=0)
        predicted_cycles[idx] = torch.max(sum)

    criterion = nn.MSELoss()

    return criterion(predicted_cycles, measured_cycles.to(device))


    # split_predictions = torch.zeros(measured_cycles.shape[0], dtype=predicted_port_pressures.dtype)
    #
    # for i in range(0, measured_cycles.shape[0]):
    #     all_max = []
    #     for idx, b in enumerate(cpu_batch):
    #         if b == i:
    #             all_max.append(torch.max(predicted_port_pressures[idx]))
    #     if len(all_max) > 0:
    #         all_max = torch.stack(all_max)
    #         split_predictions[i] = torch.sum(all_max)
    #
    # criterion = nn.MSELoss()
    #
    # return criterion(split_predictions.to(predicted_port_pressures.device), measured_cycles.to(predicted_port_pressures.device))


class Predictor(pl.LightningModule):
    def __init__(self, encoder, hidden_size, output_size, batch_size, has_virt_node=True, lr=0.001, dtype=torch.float32):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_size, output_size, dtype=dtype)
        self.relu = nn.ReLU()
        self.lr = lr
        self.batch_size = batch_size
        self.has_virt_node = has_virt_node
        #self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.fc(x)
        x = self.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        bb, measured, raw = batch
        input_sequence = bb.x.to(self.device)
        edge_index = bb.edge_index.to(self.device)
        predictions = self(input_sequence, edge_index)
        loss = loss_function(predictions, measured, bb.batch, self.has_virt_node)
        self.log("training_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        # print(self.trainer.strategy)
        # print(str(self.trainer.strategy))
        # if not str(self.trainer.strategy).find("DeepSpeed") is None:
        #     return deepspeed.ops.adam.DeepSpeedCPUAdam(model_params=self.parameters(), lr=self.lr)
        # else:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

import torch.nn as nn
import lightning.pytorch as pl
import torch
import deepspeed
from torch.functional import F
from torch_geometric.nn import Linear


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
        tensor = F.threshold(torch.stack(s).to(device), 0.2, 0)
        if has_virt_node:
            tensor = tensor[1:]

        total = tensor.sum(dim=0)
        predicted_cycles[idx] = torch.max(total)

    criterion = nn.MSELoss()

    return criterion(predicted_cycles, measured_cycles.to(device))


class Predictor(pl.LightningModule):
    def __init__(self, encoder, hidden_size, output_size, batch_size, has_virt_node=True, lr=0.001, dtype=torch.float32, l1_lambda=0.01, l2_lambda=0.01):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.fc1 = Linear(hidden_size, hidden_size // 2)
        self.fc2 = Linear(hidden_size // 2, output_size)
        self.lr = lr
        self.batch_size = batch_size
        self.has_virt_node = has_virt_node
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        #x = F.threshold(x, 0.2, 0)
        return x

    def training_step(self, batch, batch_idx):
        bb, measured, raw = batch
        input_sequence = bb.x.to(self.device)
        edge_index = bb.edge_index.to(self.device)
        predictions = self(input_sequence, edge_index)
        loss = loss_function(predictions, measured, bb.batch, self.has_virt_node)

        if self.global_step % 100 == 0:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(f"{name}_weights", param, self.global_step)
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)

        self.log("training_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        bb, measured, raw = batch
        input_sequence = bb.x.to(self.device)
        edge_index = bb.edge_index.to(self.device)
        predictions = self(input_sequence, edge_index)
        loss = loss_function(predictions, measured, bb.batch, self.has_virt_node)

        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        # print(self.trainer.strategy)
        # print(str(self.trainer.strategy))
        # if not str(self.trainer.strategy).find("DeepSpeed") is None:
        #     return deepspeed.ops.adam.DeepSpeedCPUAdam(model_params=self.parameters(), lr=self.lr)
        # else:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_lambda)

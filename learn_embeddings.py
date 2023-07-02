import torch
import lightning.pytorch as pl
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from model.utils import get_all_nodes
from torch.utils.data import DataLoader, TensorDataset
import math

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, batch_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        self.batch_size = batch_size

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        _, decoded = self(inputs)
        criterion = nn.MSELoss()
        loss = criterion(decoded, inputs)

        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss += l1_lambda * l1_norm

        self.log("training_loss", loss, on_epoch=True, batch_size=self.batch_size)

        if self.global_step % 100 == 0:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.global_step)
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)

        # if batch_idx == 0 and self.logger is not None:
        #     n_samples = min(inputs.size(0), 8)
        #     input_size = inputs.size(-1)
        #     size_approximation = int(math.sqrt(input_size))
        #     comparison = torch.cat([inputs[:n_samples], decoded[:n_samples]])
        #     self.logger.experiment.add_images("reconstructed_samples", comparison.view(n_samples * 2, 1, size_approximation, size_approximation), self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        _, decoded = self(inputs)
        criterion = nn.MSELoss()
        loss = criterion(decoded, inputs)
        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)


nodes = get_all_nodes("data/i5_1135g7.pb")

input_dim = nodes.shape[1]
hidden_dim = 256
batch_size = 32

dataset = TensorDataset(nodes)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = TensorBoardLogger("runs", name="tgl_embeddings")

model = Autoencoder(input_dim, hidden_dim, batch_size).to(device)
trainer = pl.Trainer(max_epochs=100, logger=logger, accelerator="gpu")
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

torch.save(model, "trained_models/x86_embeddings.pt")

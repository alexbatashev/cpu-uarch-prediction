import torch
import lightning.pytorch as pl
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from model.utils import BasicBlockDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GAE
from torch.functional import F
import torch.utils.data


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GCNAutoencoder(pl.LightningModule):
    def __init__(self, encoder):
        super(GCNAutoencoder, self).__init__()
        self.model = GAE(encoder)

    def forward(self, x, edge_index):
        encoded = self.model.encode(x, edge_index)
        decoded = self.model.decode(x, edge_index)

        return encoded, decoded

    def training_step(self, batch, batch_idx):
        bb, measured, raw = batch
        _, decoded = self(bb.x, bb.edge_index)

        criterion = nn.MSELoss()
        loss = criterion(decoded, bb.x)

        self.log("training_loss", loss, on_epoch=True, batch_size=self.batch_size)

        if self.global_step % 100 == 0:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.global_step)
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        bb, measured, raw = batch
        _, decoded = self(bb.x, bb.edge_index)

        criterion = nn.MSELoss()
        loss = criterion(decoded, bb.x)
        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.001)


dtype = torch.float32

dataset = BasicBlockDataset("data/i5_1135g7.pb", dtype=dtype)
print(f"Training with {len(dataset)} samples")

num_training = int(0.8 * len(dataset))
num_val = len(dataset) - num_training

print(f"Split into {num_training} training and {num_val} validation samples")

input_dim = dataset.num_opcodes
hidden_dim = 256
batch_size = 128

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_training, num_val])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = TensorBoardLogger("runs", name="tgl_embeddings")

encoder = GCNEncoder(input_dim, hidden_dim)
model = GCNAutoencoder(encoder).to(device)
trainer = pl.Trainer(max_epochs=2, logger=logger, accelerator="gpu")
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

torch.save(model, "trained_models/x86_embeddings_graph.pt")

model.eval()

sample1, _, _ = val_dataset[120]
_, decoded = model(sample1.x, sample1.edge_index)
print(sample1.x)
print(torch.argmax(sample1.x, dim=1))
print(decoded)
print(torch.argmax(decoded, dim=1))
print(torch.max(decoded, dim=1))

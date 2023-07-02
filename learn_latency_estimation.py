import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, GATConv, RGCNConv
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from model.utils import BasicBlockDataset, plot_hist
import torch.utils.data
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import PIL.Image
from torchvision.transforms import ToTensor


class GNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, learning_rate=0.02):
        super(GNN, self).__init__()
        self.heads = 6
        self.gat = GATConv(input_dim, hidden_dim, heads=self.heads)
        self.conv1 = RGCNConv(hidden_dim * self.heads, hidden_dim, 1)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.lr = learning_rate
        self.batch_size = batch_size
        self.val_measurements = [[], []]

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat(x, edge_index, data.edge_attr)
        x = F.relu(x)

        x = self.conv1(x, edge_index, data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling to aggregate graph features
        x = global_add_pool(x, batch)
        x = self.fc(x)
        x = F.relu(x)
        return x.squeeze(-1)

    def training_step(self, batch, batch_idx):
        bb, measured, _ = batch
        y_hat = self(bb)
        loss = F.mse_loss(y_hat, measured)
        l1_reg = 0.0
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
        loss = loss + 1e-5 * l1_reg  # 1e-5 is the L1 penalty coefficient
        mape = torch.mean(torch.abs((measured - y_hat) / measured)) * 100
        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_mape", mape, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        bb, measured, raw = batch
        y_hat = self(bb)
        loss = F.mse_loss(y_hat, measured)
        mape = torch.mean(torch.abs((measured - y_hat) / measured)) * 100
        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("val_mape", mape, on_epoch=True, batch_size=self.batch_size)

        for a, b in zip(y_hat, measured):
            self.val_measurements[0].append(a.item())
            self.val_measurements[1].append(b.item())

        if batch_idx == 0:
            num_samples_to_log = 5
            for i in range(num_samples_to_log):
                self.logger.experiment.add_scalar(f"val/sample_{i}/true", measured[i].item(), self.current_epoch)
                self.logger.experiment.add_scalar(f"val/sample_{i}/predicted", y_hat[i].item(), self.current_epoch)
                if self.global_step == 0:
                    self.logger.experiment.add_text(f"val/sample_{i}/source", raw['source'][i], self.global_step)

    def on_validation_epoch_end(self):
        plot = plot_hist(np.array(self.val_measurements))
        image = PIL.Image.open(plot)
        image = ToTensor()(image).unsqueeze(0)
        self.logger.experiment.add_image("val_histogram", image[0], self.current_epoch)
        self.val_measurements = [[], []]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


def main():
    dtype = torch.float32
    batch_size = 32
    hidden_dim = 64
    output_dim = 1

    dataset = BasicBlockDataset("data/i5_1135g7_v8.pb", dtype=dtype)
    print(f"Training with {len(dataset)} samples")

    num_training = int(0.8 * len(dataset))
    num_val = len(dataset) - num_training

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_training, num_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    model = GNN(dataset.num_opcodes, hidden_dim, output_dim, batch_size)

    logger = TensorBoardLogger("runs", name="tgl_estimate")
    trainer = pl.Trainer(max_epochs=100, logger=logger)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
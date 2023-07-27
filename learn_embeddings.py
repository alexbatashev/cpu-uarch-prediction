from python.llvm_ml.data import load_pyg_dataset
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from python.llvm_ml.utils import plot_histogram
import torch.utils.data
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import PIL.Image
from torchvision.transforms import ToTensor
from model.estimation import GNNEstimation, LSTMEstimation
from torch.nn import Module, Linear, LSTM
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.utils import to_dense_batch
import torch

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

dataset = load_pyg_dataset("./data/ryzen3600_v1.pb", prefilter=False)
print(f"Training with {len(dataset)} samples")

class Embedding(pl.LightningModule):
    def __init__(self, input_dim, parameters):
        super().__init__()
        self.hidden_dim = parameters['hidden_dim']
        self.batch_size = parameters['batch_size']

        self.conv = GCNConv(input_dim, self.hidden_dim)

        # self.embedding = nn.Embedding(20000, hidden_dim)

        self.lstm1 = LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.lstm2 = LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.hidden1 = self.init_hidden(self.hidden_dim)
        self.hidden2 = self.init_hidden(self.hidden_dim)

        self.fc = Linear(self.hidden_dim, input_dim)

        self.lr = parameters['learning_rate']

    def init_hidden(self, out_dim):
        return torch.zeros([1, self.batch_size, out_dim]).cuda().detach(), torch.zeros([1, self.batch_size, out_dim]).cuda().detach()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv(x, edge_index)
        x = F.relu(x)

        nodes, mask = to_dense_batch(x, batch)

        # x = self.embedding(nodes[:, 1:, :].type(torch.LongTensor).cuda())

        x, hidden = self.lstm1(nodes[:, 1:, :], self.hidden1)
        h0, h1 = hidden
        self.hidden1 = (h0.detach(), h1.detach())

        #x = torch.flip(x, dims=[0, 1])

        # x, hidden = self.lstm2(x, self.hidden2)
        # h0, h1 = hidden
        # self.hidden2 = (h0.detach(), h1.detach())

        #x = torch.flip(x, dims=[0, 1])
        # x = F.relu(x)

        x = self.fc(x)
        x = F.sigmoid(x)

        return x

    def training_step(self, batch, batch_idx):
        bb, raw = batch
        y_hat = self(bb)

        nodes, mask = to_dense_batch(bb.x, bb.batch)

        loss = F.cross_entropy(y_hat, nodes[:, 1:, :])
        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        bb, raw = batch
        y_hat = self(bb)

        nodes, mask = to_dense_batch(bb.x, bb.batch)

        loss = F.cross_entropy(y_hat, nodes[:, 1:, :])
        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss

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


parameters = {
    'batch_size': 64,
    'hidden_dim': 128,
    'learning_rate': 0.1,
}

num_training = int(0.8 * len(dataset))
num_val = len(dataset) - num_training

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_training, num_val])
train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, num_workers=6, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=False, num_workers=6, drop_last=True)

model = Embedding(16, parameters)

logger = TensorBoardLogger("runs", name="x64_embedding")
logger.log_graph(model)
logger.log_hyperparams(parameters)
trainer = pl.Trainer(max_epochs=100, logger=logger)
trainer.fit(model, train_loader, val_loader)
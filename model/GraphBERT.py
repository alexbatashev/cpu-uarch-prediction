import lightning.pytorch as pl
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam, lr_scheduler


class MCEmbedding(pl.LightningModule):
    def __init__(self, num_opcodes, emb_size):
        super().__init__()

        self.embedding = nn.Embedding(num_opcodes, emb_size)
        self.pos_encoding = gnn.PositionalEncoding(emb_size)
        self.norm = gnn.LayerNorm(emb_size)

    def forward(self, input_tensor):
        pos_tensor = self.pos_encoding(input_tensor)

        output = self.embedding(input_tensor) + pos_tensor

        return self.norm(output)


class MCEncoder(pl.LightningModule):
    def __init__(self, emb_size, out_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = gnn.GATConv(emb_size, out_size, heads=num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, out_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(out_size, emb_size),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(emb_size)

    def forward(self, nodes, edge_index, batch):
        context = self.attention(nodes, edge_index)
        dense_context = to_dense_batch(context, batch)
        res = self.feed_forward(dense_context)

        return self.norm(res)


class ThroughputEstimator(pl.LightningModule):
    def __init__(self, num_opcodes, emb_size, batch_size, hidden_size=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = MCEmbedding(num_opcodes, emb_size)
        self.encoder = MCEncoder(emb_size, hidden_size, num_heads, dropout)
        self.token_prediction = nn.Linear(emb_size, num_opcodes)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.regression = nn.Linear(emb_size, 1)

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, nodes, edge_index, batch):
        embedded = self.embedding(nodes)
        encoded = self.encoder(embedded)

        token_predictions = self.token_prediction(encoded)

        first_token = encoded[:, 0, :]

        return self.softmax(token_predictions), self.regression(first_token)

    def training_step(self, batch, batch_idx):
        bb, raw = batch

        masked_token, y_hat = self.forward(bb.x, bb.edge_index, bb.batch)

        y_hat = y_hat.reshape(bb.y.shape)

        reg_loss = F.mse_loss(y_hat, bb.y)

        loss = reg_loss  # + ml_loss

        self.train_mae(y_hat, bb.y)

        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_mae", self.train_mae, on_epoch=True, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        bb, raw = batch

        masked_token, y_hat = self.forward(bb.x, bb.edge_index, bb.batch)

        y_hat = y_hat.reshape(bb.y.shape)

        reg_loss = F.mse_loss(y_hat, bb.y)

        loss = reg_loss  # + ml_loss

        self.val_mae(y_hat, bb.y)

        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("val_mae", self.train_mae, on_epoch=True, batch_size=self.batch_size)

        if batch_idx == 0:
            num_samples_to_log = 5
            for i in range(num_samples_to_log):
                self.logger.experiment.add_scalar(f"val/sample_{i}/true", bb.y[i].item(), self.current_epoch)
                self.logger.experiment.add_scalar(f"val/sample_{i}/predicted", y_hat[i].item(), self.current_epoch)
                if self.global_step == 0:
                    self.logger.experiment.add_text(f"val/sample_{i}/source", raw['source'][i], self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
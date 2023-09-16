import lightning.pytorch as pl
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_batch, to_dense_adj, one_hot
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam, lr_scheduler
from torch.nn import Module
import torch


class MCEmbedding(Module):
    def __init__(self, num_opcodes, emb_size):
        super().__init__()

        self.embedding = nn.Embedding(num_opcodes, emb_size)
        self.pos_encoding = gnn.PositionalEncoding(emb_size)
        self.norm = gnn.LayerNorm(emb_size)

    def forward(self, input_tensor):
        pos_tensor = self.pos_encoding(input_tensor)

        output = self.embedding(input_tensor) + pos_tensor

        return self.norm(output)


class MCGraphAttention(Module):
    def __init__(self, emb_size, out_size, num_heads):
        super().__init__()

        self.attention = gnn.DenseGATConv(emb_size, out_size, heads=num_heads)
        self.linear = nn.Linear(out_size * num_heads, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, input_tensor, edge_index):
        dense_scores = self.attention(input_tensor, edge_index)

        dense_scores = self.linear(dense_scores)

        return self.norm(dense_scores)


class MCGraphEncoder(Module):
    def __init__(self, emb_size, out_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = MCGraphAttention(emb_size, out_size, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, out_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(out_size, emb_size),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(emb_size)

    def forward(self, nodes, edge_index):
        dense_context = self.attention(nodes, edge_index)

        res = self.feed_forward(dense_context)

        return self.norm(res)


class ThroughputEstimator(pl.LightningModule):
    def __init__(self, num_opcodes, emb_size, batch_size, hidden_size=256, num_heads=4, dropout=0.1, learning_rate=0.01):
        super().__init__()
        self.embedding = MCEmbedding(num_opcodes, emb_size)

        self.encoders = nn.ModuleList([MCGraphEncoder(emb_size, hidden_size, num_heads, dropout) for _ in range(4)])

        self.token_prediction = nn.Linear(emb_size, num_opcodes)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.regression = nn.Linear(emb_size, 1)

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_opcodes = num_opcodes

    def forward(self, nodes, edge_index, batch):
        embedded = self.embedding(nodes)

        encoded, _ = to_dense_batch(embedded, batch)

        dense_edges = to_dense_adj(edge_index, batch)

        for encoder in self.encoders:
            encoded = encoder(encoded, dense_edges)

        token_predictions = self.token_prediction(encoded)

        first_token = encoded[:, 0, :]

        return self.softmax(token_predictions), self.regression(first_token)

    def training_step(self, batch, batch_idx):
        bb, raw, mask_id, original_token = batch

        masked_token, y_hat = self.forward(bb.x, bb.edge_index, bb.batch)

        dense_x, _ = to_dense_batch(bb.x, bb.batch)

        target_token = torch.zeros(masked_token.shape)

        for i in range(masked_token.shape[0]):
            for j in range(masked_token.shape[1]):
                target_token[i, j, dense_x[i, j]] = 1

        for i in range(self.batch_size):
            if mask_id[i] != 0:
                target_token[i, mask_id[i], original_token[i]] = 1

        ml_loss = F.cross_entropy(masked_token, target_token)

        y_hat = y_hat.reshape(bb.y.shape)

        reg_loss = F.mse_loss(y_hat, bb.y)

        loss = reg_loss + ml_loss

        self.train_mae(y_hat, bb.y)

        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_ml_loss", ml_loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_reg_loss", reg_loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_mae", self.train_mae, on_epoch=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        bb, raw, mask_id, original_token = batch

        masked_token, y_hat = self.forward(bb.x, bb.edge_index, bb.batch)

        dense_x, _ = to_dense_batch(bb.x, bb.batch)

        target_token = torch.zeros(masked_token.shape)

        for i in range(masked_token.shape[0]):
            for j in range(masked_token.shape[1]):
                target_token[i, j, dense_x[i, j]] = 1

        for i in range(self.batch_size):
            if mask_id[i] != 0:
                target_token[i, mask_id[i], original_token[i]] = 1

        ml_loss = F.cross_entropy(masked_token, target_token)

        y_hat = y_hat.reshape(bb.y.shape)

        reg_loss = F.mse_loss(y_hat, bb.y)

        loss = reg_loss + ml_loss

        self.val_mae(y_hat, bb.y)

        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("val_mae", self.val_mae, on_epoch=True, batch_size=self.batch_size)

        if batch_idx == 0:
            num_samples_to_log = 5
            for i in range(num_samples_to_log):
                self.logger.experiment.add_scalar(f"val/sample_{i}/true", bb.y[i].item(), self.current_epoch)
                self.logger.experiment.add_scalar(f"val/sample_{i}/predicted", y_hat[i].item(), self.current_epoch)
                if self.global_step == 0:
                    self.logger.experiment.add_text(f"val/sample_{i}/source", raw['source'][i], self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1, 7, 10, 15, 25, 30], gamma=0.1, verbose=False)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss',
        #     }
        # }

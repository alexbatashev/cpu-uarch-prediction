import lightning.pytorch as pl
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import to_dense_batch, to_dense_adj, scatter
import torch.nn.functional as F
import torchmetrics
from torch.optim import Adam
from torch.nn import Module
from torch.optim import lr_scheduler
import torch


def to_dense_adj_batch(edge_index, batch, num_nodes=0):
    batch_size = int(batch.max()) + 1
    adj = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.int64, device=edge_index.device)

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')

    print(num_nodes)

    return adj


class MCEmbedding(Module):
    def __init__(self, num_opcodes, emb_size):
        super().__init__()

        self.embedding = nn.Embedding(num_opcodes, emb_size)
        self.pos_encoding = gnn.PositionalEncoding(emb_size)
        self.norm = gnn.LayerNorm(emb_size)

    def forward(self, input_tensor):
        pos_tensor = self.pos_encoding(input_tensor)

        output = self.embedding(input_tensor) + pos_tensor

        return self.norm(output), pos_tensor


class MCAttentionHead(nn.Module):
    def __init__(self, in_size, heads=4):
        super().__init__()

        self.num_heads = heads

        self.key = nn.Linear(in_size, in_size, bias=False)
        self.query = nn.Linear(in_size, in_size, bias=False)
        self.value = nn.Linear(in_size, in_size, bias=False)

        self.proj = nn.Linear(in_size, in_size, bias=False)

    def forward(self, nodes, edge_index, mask=None):
        B, T, C = nodes.shape

        mask = mask.view(B, T, 1)
        mask = torch.broadcast_to(mask, (B, T, C))

        nodes = nodes.masked_fill(mask == False, float(0))
        # for b in range(B):
        #     for idx in range(mask.shape[1]):
        #         if mask[b, idx] is False:
        #             nodes[b, idx, :] = 0

        k = self.key(nodes).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = self.query(nodes).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        weight = torch.matmul(q, k.transpose(-2, -1)) * self.num_heads ** -0.5

        scale = 3 * edge_index + torch.ones(edge_index.shape, device=edge_index.device)

        scale = torch.cat([torch.cat([scale[x] for _ in range(self.num_heads)]) for x in range(scale.shape[0])]).view(B, self.num_heads, T, T)

        weight = weight * scale

        # if mask is not None:
        #     for b in range(mask.shape[0]):
        #         for n in range(mask.shape[1]):
        #             if mask[b][n] is False:
        #                 weight[b, :, n, :] = float('-inf')
        #                 weight[b, :, :, n] = float('-inf')

        # TODO for decoder weight must be a triangle matrix
        weight = F.softmax(weight, dim=-1)

        v = self.value(nodes).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        res = torch.matmul(weight, v)

        res = res.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(res)


class MCGraphAttention(nn.Module):
    def __init__(self, in_size, heads=6):
        super().__init__()

        # self.heads = nn.ModuleList([MCAttentionHead(in_size, heads) for _ in range(heads)])
        self.heads = MCAttentionHead(in_size, heads)

    def forward(self, nodes, edge_index, mask=None):
        # return torch.cat([h(nodes, edge_index, mask) for h in self.heads], dim=-1)
        return self.heads(nodes, edge_index, mask)

# class MCGraphAttention(gnn.conv.MessagePassing):
#     def __init__(self, emb_size, hidden_size, num_heads):
#         super().__init__()
#
#         self.lin_key = nn.Linear(emb_size, num_heads * emb_size)
#         self.lin_query = nn.Linear(emb_size, num_heads * emb_size)
#         self.lin_value = nn.Linear(emb_size, num_heads * emb_size)
#
#         self.lin_skip = nn.Linear(emb_size, num_heads * emb_size)
#
#         self.num_heads = num_heads
#         self.emb_size = emb_size
#
#     def forward(self, input_tensor, edge_index, mask):
#         H, C = self.num_heads, self.emb_size
#
#         query = self.lin_query(input_tensor[1]).view(-1, H, C)
#         key = self.lin_key(input_tensor[0]).view(-1, H, C)
#         value = self.lin_value(input_tensor[0]).view(-1, H, C)
#
#         out = self.propagate(edge_index, query=query, key=key, value=value, size=None)
#
#         return out.view(-1, H * C)


class MCGraphEncoder(Module):
    def __init__(self, emb_size, out_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.norm1 = gnn.LayerNorm(emb_size)
        self.attention = MCGraphAttention(emb_size, heads=8)
        self.norm2 = gnn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

        # self.attention = MCGraphAttention(emb_size, out_size, num_heads)
        #
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(emb_size, out_size),
        #     nn.Dropout(dropout),
        #     nn.GELU(),
        #     nn.Linear(out_size, emb_size),
        #     nn.Dropout(dropout)
        # )
        #
        # self.norm = nn.LayerNorm(emb_size)

    def forward(self, nodes, edge_index, mask):
        nodes = self.norm1(nodes)
        dense_context = self.attention(nodes, edge_index, mask)

        res = self.feed_forward(dense_context)

        return self.norm2(res)


class ThroughputEstimator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        if 'mode' not in config:
            config['mode'] = 'regression'
        if 'hidden_size' not in config:
            config['hidden_size'] = 64
        if 'dropout' not in config:
            config['dropout'] = 0.1
        if 'learning_rate' not in config:
            config['learning_rate'] = 1e-5
        if 'num_heads_encoder' not in config:
            config['num_heads_encoder'] = 4
        if 'num_heads_decoder' not in config:
            config['num_heads_decoder'] = 4
        if 'num_encoders' not in config:
            config['num_encoders'] = 8

        if config['mode'] != 'pretrain' and config['mode'] != 'regression':
            raise Exception("Configuration exception")

        self.config = config

        num_opcodes = config['num_opcodes']
        emb_size = config['embedding_size']
        num_heads_encoder = config['num_heads_encoder']
        num_encoders = config['num_encoders']
        hidden_size = config['hidden_size']
        dropout = config['dropout']

        self.embedding = MCEmbedding(num_opcodes, emb_size)

        self.encoders = nn.ModuleList(
           [MCGraphEncoder(emb_size, hidden_size, num_heads_encoder, dropout) for _ in range(num_encoders)])

        self.token_prediction = nn.Linear(emb_size, num_opcodes)

        self.proj = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.Linear(4 * emb_size, emb_size)
        )

        self.regression = nn.Sequential(
           nn.Linear(emb_size, 1)
        )

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, nodes, edge_index, batch):
        embedded, pos_enc = self.embedding(nodes)

        encoded, mask = to_dense_batch(embedded, batch)

        # TODO this does not do what I expected
        dense_edges = to_dense_adj(edge_index, batch)
        dense_edges = dense_edges.view(encoded.shape[0], encoded.shape[1], encoded.shape[1])

        for encoder in self.encoders:
            encoded = encoder(encoded, dense_edges, mask)

        token_predictions = self.token_prediction(encoded)

        encoded = self.proj(encoded)

        x = self.regression(encoded[:, 0, :])

        return token_predictions, x

    def _step(self, batch, stage: str):
        assert (stage == 'train') or (stage == 'val')

        bb, raw, mask_id, original_token = batch

        masked_token, y_hat = self.forward(bb.x, bb.edge_index, bb.batch)

        log_prefix = "train" if stage == 'train' else "val"

        if self.config['mode'] == 'pretrain':

            dense_x, _ = to_dense_batch(bb.x, bb.batch)

            target_token = dense_x.clone()

            for i in range(self.config['batch_size']):
                if mask_id[i] != 0:
                    target_token[i, mask_id[i]] = original_token[i]

            loss = F.cross_entropy(masked_token.view(-1, self.config['num_opcodes']), target_token.view(-1).long())
        else:
            y_hat = y_hat.reshape(bb.y.shape)

            loss = F.huber_loss(y_hat, bb.y)

            mse = F.mse_loss(y_hat, bb.y)

            mae_metric = self.train_mae if stage == 'train' else self.val_mae
            mae_metric(y_hat, bb.y)

            self.log(f"{log_prefix}_mae", mae_metric, on_epoch=True, batch_size=self.config['batch_size'])
            self.log(f"{log_prefix}_mse", mse, on_epoch=True, batch_size=self.config['batch_size'])

        self.log(f"{log_prefix}_loss", loss, on_epoch=True, batch_size=self.config['batch_size'])

        return loss, y_hat, bb, raw

    def training_step(self, batch, batch_idx):
        loss, y_hat, bb, raw = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, bb, raw = self._step(batch, 'val')

        if batch_idx == 0:
            num_samples_to_log = 5
            for i in range(num_samples_to_log):
                self.logger.experiment.add_scalar(f"val/sample_{i}/true", bb.y[i].item(), self.current_epoch)
                self.logger.experiment.add_scalar(f"val/sample_{i}/predicted", y_hat[i].item(), self.current_epoch)
                if self.global_step == 0:
                    self.logger.experiment.add_text(f"val/sample_{i}/source", raw['source'][i], self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=1e-3)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

        if self.config['mode'] == 'pretrain':
            self.regression.requires_grad_(False)
        else:
            self.token_prediction.requires_grad_(False)
            self.embedding.requires_grad_(False)
            self.encoders.requires_grad_(False)
    #
    # def on_validation_epoch_end(self):
    #     if not self.debug:
    #         # x = np.asarray(self.val_measurements[0])
    #         # y = np.asarray(self.val_measurements[1])
    #         # if np.any(x) and np.any(y):
    #         #    plot = plot_histogram(x, y, percentile=0.95)
    #         #    image = PIL.Image.open(plot)
    #         #    image = ToTensor()(image).unsqueeze(0)
    #         #    self.logger.experiment.add_image("val_histogram", image[0], self.current_epoch)
    #
    #         lift_chart = plot_lift_chart(self.val_measurements)
    #         image = PIL.Image.open(lift_chart)
    #         image = ToTensor()(image).unsqueeze(0)
    #         self.logger.experiment.add_image("val_lift_chart", image[0], self.current_epoch)
    #     self.val_measurements = [[], []]
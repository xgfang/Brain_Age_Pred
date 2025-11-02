import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class BrainAgeTransformer(nn.Module):
    def __init__(
        self,
        n_features,        # number of latent VBM features per subject
        d_model   = 64,
        nhead     = 8,
        num_layers= 3,
        ff_dim    = 128,
        dropout   = 0.1
    ):
        super().__init__()
        # project each scalar feature → d_model
        self.input_proj = nn.Linear(1, d_model)
        # add positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=n_features)
        # apply transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # apply linear layer to get age prediction
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch_size, n_features), dtype=torch.float
        returns: (batch_size,) predicted age
        """
        # turn into "sequence" of scalar tokens
        x = x.unsqueeze(-1)                  # → (B, L, 1)
        x = self.input_proj(x)              # → (B, L, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)                   # mean‑pool over features
        age = self.head(x).squeeze(-1)      # → (B,)
        return age


# --- training loop sketch ---
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    for feats, ages in loader:
        feats, ages = feats.to(device), ages.to(device)
        pred = model(feats)
        loss = loss_fn(pred, ages)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)

# def eval_epoch(model, loader, loss_fn, device):
#     model.eval()
#     total, mae = 0, 0
#     with torch.no_grad():
#         for feats, ages in loader:
#             feats, ages = feats.to(device), ages.to(device)
#             pred = model(feats)
#             mae += (pred - ages).abs().sum().item()
#             total += feats.size(0)
#     return mae / total

def eval_epoch(model, loader, device):
    model.eval()
    total_se = 0.0
    total_n  = 0
    with torch.no_grad():
        for feats, ages in loader:
            feats, ages = feats.to(device), ages.to(device)
            pred = model(feats)                           # (B,)
            se   = F.mse_loss(pred, ages, reduction="sum")  # sum of squared errors
            total_se += se.item()
            total_n  += feats.size(0)
    return total_se / total_n  # this is the mean squared error

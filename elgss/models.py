"""Policy networks: ERQ-Net (GAT), GCN, single-head GAT, MLP, and a
Transformer set-encoder (the non-graph attention baseline)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

N_ACTIONS = 5


class GATPolicy(nn.Module):
    """ERQ-Net: 2-layer GAT, 6->128 hidden, 4 heads.

    NOTE: uses the standard additive-attention GATConv (Velickovic et al.,
    2017). The static-attention limitation of this operator is discussed by
    Brody et al. (GATv2, ICLR 2022); swapping GATConv for GATv2Conv is a
    drop-in change if you wish to test dynamic attention.
    """
    def __init__(self, in_dim=6, edge_dim=8, hid=128, heads=4):
        super().__init__()
        self.g1 = GATConv(in_dim, hid, heads=heads, edge_dim=edge_dim)
        self.g2 = GATConv(hid * heads, hid, heads=1, edge_dim=edge_dim)
        self.head = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(),
                                  nn.Linear(64, N_ACTIONS))
        self.last_attention = None  # (edge_index, alpha) for analysis

    def forward(self, b, keep_attention=False):
        if keep_attention:
            h, (ei, a) = self.g1(b.x, b.edge_index, b.edge_attr,
                                 return_attention_weights=True)
            self.last_attention = (ei.detach(), a.detach())
            h = F.elu(h)
        else:
            h = F.elu(self.g1(b.x, b.edge_index, b.edge_attr))
        h = F.elu(self.g2(h, b.edge_index, b.edge_attr))
        return self.head(h[b.ptr[:-1]])  # ego node per graph


class GAT1Head(GATPolicy):
    def __init__(self):
        super().__init__(heads=1)


class GCNPolicy(nn.Module):
    def __init__(self, in_dim=6, hid=128):
        super().__init__()
        self.g1, self.g2 = GCNConv(in_dim, hid), GCNConv(hid, hid)
        self.head = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(),
                                  nn.Linear(64, N_ACTIONS))

    def forward(self, b, **kw):
        h = F.elu(self.g1(b.x, b.edge_index))
        h = F.elu(self.g2(h, b.edge_index))
        return self.head(h[b.ptr[:-1]])


class MLPPolicy(nn.Module):
    """Flattened 15x7 kinematics -> MLP."""
    def __init__(self, in_dim=105):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, N_ACTIONS))

    def forward(self, flat_obs, **kw):
        return self.net(flat_obs)


class SetTransformerPolicy(nn.Module):
    """2-layer TransformerEncoder over vehicle tokens with a presence mask;
    ego-token readout. The standard non-graph attention alternative."""
    def __init__(self, in_dim=7, d=128, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=256,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                  nn.Linear(64, N_ACTIONS))

    def forward(self, obs_batch, **kw):
        mask = obs_batch[..., 0] < 0.5  # True = pad
        h = self.enc(self.embed(obs_batch), src_key_padding_mask=mask)
        return self.head(h[:, 0])  # ego token

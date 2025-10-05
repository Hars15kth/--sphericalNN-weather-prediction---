# models/equivariant_point_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, FullyConnectedTensorProduct

class EquivariantPointBlock(nn.Module):
    """
    Message passing from scalar node features and relative vectors.
    Uses tensor product to mix inputs into hidden representation,
    followed by manual gating to produce equivariant output.
    """

    def __init__(self, irreps_in: Irreps, irreps_hidden: Irreps, irreps_out: Irreps):
        super().__init__()
        self.tp1 = FullyConnectedTensorProduct(irreps_in, Irreps("1x1o"), irreps_hidden)

        self.lin_hidden = nn.Linear(irreps_hidden.dim, irreps_hidden.dim)
        self.lin_out = nn.Linear(irreps_hidden.dim, irreps_out.dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(irreps_hidden.dim, irreps_hidden.dim),
            nn.SiLU(),
            nn.Linear(irreps_hidden.dim, irreps_out.dim)
        )

        self.lift = nn.Sequential(
            nn.Linear(irreps_in.dim, irreps_in.dim),
            nn.SiLU(),
            nn.Linear(irreps_in.dim, irreps_in.dim)
        )

    def forward(self, feats_in: torch.Tensor, rel_vec: torch.Tensor):
        """
        feats_in: [B, N, irreps_in.dim] — scalar node features
        rel_vec:  [B, N, K, 3] — relative vectors (treated as 1o)
        """
        B, N, K, _ = rel_vec.shape
        x = self.lift(feats_in)  # [B, N, irreps_in.dim]
        x = x.unsqueeze(2).expand(B, N, K, x.shape[-1]).reshape(B, N * K, -1)
        v = rel_vec.reshape(B, N * K, 3)

        h = self.tp1(x, v)  # [B, N*K, irreps_hidden.dim]
        h = F.silu(self.lin_hidden(h))
        gates = torch.sigmoid(self.gate_mlp(h))  # [B, N*K, irreps_out.dim]
        out = self.lin_out(h) * gates  # gated output

        return out.reshape(B, N, K, -1)
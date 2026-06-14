import torch
import torch.nn as nn
from torch import Tensor


class GraphEvidenceTokenizer(nn.Module):
    """Compresses center-frame edge set E_c into M fixed LLM tokens.

    Input:
        E_c:        (B, N, Tl, De)  — center frame edge tensor, Tl = N+2
        edge_valid: (B, N, 2N+2)   — boolean mask from GazeGraphBlock

    Output:
        G_LLM: (B, M, d_llm)
    """

    def __init__(self, edge_dim: int, d_llm: int, m: int = 32,
                 depth: int = 1, num_heads: int = 8):
        super().__init__()
        self.m = m
        self.queries = nn.Parameter(torch.randn(m, edge_dim) * 0.02)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(edge_dim, num_heads, batch_first=True)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(edge_dim) for _ in range(depth)])
        self.proj = nn.Linear(edge_dim, d_llm)

    def forward(self, E_c: Tensor, edge_valid: Tensor) -> Tensor:
        B, N, Tl, De = E_c.shape
        assert edge_valid.shape == (B, N, 2 * N + 2), (
            f"edge_valid shape mismatch: expected {(B, N, 2*N+2)}, got {tuple(edge_valid.shape)}"
        )

        # Build valid mask matching E_c's Tl = N+2
        #   edge_valid: (B, N, 2N+2)  [0:N]=p2p, [N:2N]=SA-proxy, [2N:2N+2]=nulls
        #   We need: [0:N]=p2p, [N]=null_in, [N+1]=null_out  → (B, N, Tl)
        ev_tl = torch.cat([
            edge_valid[:, :, :N],        # (B, N, N)  person targets
            edge_valid[:, :, 2 * N:],    # (B, N, 2)  null_in + null_out
        ], dim=2)                        # (B, N, Tl)

        kv = E_c.reshape(B, N * Tl, De)           # (B, N*Tl, De)
        kpm = ~ev_tl.reshape(B, N * Tl)           # True = masked out (B, N*Tl)

        # When all keys are masked, MultiheadAttention returns NaN.
        # Fall back to unmasked (let attention spread uniformly) for fully-padded clips.
        all_masked = kpm.all(dim=1, keepdim=True)  # (B, 1)
        kpm = kpm & ~all_masked                    # un-mask when nothing is valid

        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, De)
        for attn, norm in zip(self.attn_layers, self.norms):
            delta, _ = attn(q, kv, kv, key_padding_mask=kpm)
            q = norm(q + delta)

        return self.proj(q)  # (B, M, d_llm)

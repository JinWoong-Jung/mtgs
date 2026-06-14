# tests/test_graph_tokenizer.py
import torch
import pytest
from mtgs.networks.vlm.graph_tokenizer import GraphEvidenceTokenizer

def _make_inputs(B=2, T=5, N=4, De=128):
    Tl = N + 2
    E = torch.randn(B, T, N, Tl, De)
    # edge_valid: (B, N, 2N+2), mark all p2p valid except self-loops, nulls valid
    ev = torch.zeros(B, N, 2 * N + 2, dtype=torch.bool)
    for i in range(N):
        for j in range(N):
            if i != j:
                ev[:, i, j] = True   # p2p
                ev[:, i, N + j] = True  # SA proxy
        ev[:, i, 2 * N] = True      # null_in
        ev[:, i, 2 * N + 1] = True  # null_out
    return E, ev

def test_output_shape():
    B, De, M, d_llm = 2, 128, 32, 2560
    tok = GraphEvidenceTokenizer(edge_dim=De, d_llm=d_llm, m=M)
    E, ev = _make_inputs(B=B, De=De)
    G = tok(E[:, E.shape[1] // 2], ev)
    assert G.shape == (B, M, d_llm), f"Expected ({B},{M},{d_llm}), got {G.shape}"

def test_all_padding_does_not_crash():
    """Clip where all edges are invalid (fully padded) should not crash."""
    B, De, M, d_llm = 1, 128, 32, 2560
    tok = GraphEvidenceTokenizer(edge_dim=De, d_llm=d_llm, m=M)
    N, Tl = 4, 6
    E_c = torch.randn(B, N, Tl, De)
    ev = torch.zeros(B, N, 2 * N + 2, dtype=torch.bool)  # all masked
    G = tok(E_c, ev)
    assert G.shape == (B, M, d_llm)

def test_variable_n():
    """N=11 (max_people=11 config) should work."""
    B, De, M, d_llm = 1, 128, 32, 2560
    tok = GraphEvidenceTokenizer(edge_dim=De, d_llm=d_llm, m=M)
    E, ev = _make_inputs(B=B, N=11, De=De)
    G = tok(E[:, E.shape[1] // 2], ev)
    assert G.shape == (B, M, d_llm)

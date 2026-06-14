# LLM Reasoner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Evidence-Augmented LLM Reasoner (Qwen3.5-4B + cross-attn) on top of frozen gaze_graph to improve LAH/LAEO/SA AP — without touching the existing gazefollow/vsgaze training pipeline.

**Architecture:** Frozen MTGS+GazeGraphBlock produces edge states (E, v_src, v_tgt) captured via forward hook on `_UnifiedRefiner`. A `GraphEvidenceTokenizer` compresses center-frame edge features into M=32 LLM tokens (G_LLM). `EvidenceAugmentedLLM` wraps Qwen3.5-4B (frozen) — injecting cross-attn into its 8 Gated Attention layers — and answers Yes/No queries with person identity grounded via node features.

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra, HuggingFace Transformers (`Qwen/Qwen3.5-4B`, `AutoModelForCausalLM`, `AutoTokenizer`)

---

## Constraints (read before touching any file)

- **Do NOT modify** `mtgs/networks/adaptor_modules.py`, `mtgs/networks/mtgs_net.py`, `mtgs/networks/models.py`, `mtgs/datasets/vsgaze.py`, `scripts/main.py`, or any existing shell scripts.
- `mtgs/config/config.yaml` gets a new `interaction.llm` block appended — existing keys are untouched.
- All new Python code lives in `mtgs/networks/llm/` or `mtgs/datasets/gaze_qa.py`.
- Capture of E/v_src/v_tgt uses a **forward hook on `_UnifiedRefiner`** — zero changes to GazeGraphBlock.

## Key conventions from existing codebase

```
Batch shapes (after pad_collate_fn, N = max people in batch):
  person_tokens:     (B, T, N, D)         D=512 (4×128 concat)
  lah_labels:        (B, T, P)            P = N*(N-1) directed pairs
  laeo_labels:       (B, T, P)
  coatt_labels:      (B, T, P)
  num_valid_people:  (B, 1)
  head_bboxes:       (B, T, N, 4)

_UnifiedRefiner.forward() returns: (E, v_src, v_tgt)
  E:     (B, T, N, Tl, De)   Tl=N+2, De=cfg.interaction.gaze_graph.edge_dim
  v_src: (B, T, N, De)
  v_tgt: (B, T, Tl, De)

GazeGraphBlock.forward() returns 6-tuple (last is edge_valid):
  edge_valid: (B, N, 2N+2)
    [0:N]    = p2p valid
    [N:2N]   = SA proxy (same as p2p)
    [2N]     = null_in valid
    [2N+1]   = null_out valid

Pair ordering for labels[b, t, k]:
  pairs = list(itertools.permutations(range(N), 2))
  pairs[k] = (src_k, dst_k)
  Dataset convention: lah_labels[b,t,k] = 1 means person src_k looks at dst_k
  (confusingly named: src IS the looker, dst IS the target)
  → LAH query: "Does <P_{src}> look at <P_{dst}>?"

Valid persons: right-aligned in padded N.
  valid_start = N - int(num_valid_people[b])
  valid indices: [valid_start, N)

Center frame: t_c = T // 2  (T=5 → t_c=2, matches models.py middle_frame_idx)
```

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `mtgs/networks/llm/__init__.py` | **Create** | package init |
| `mtgs/networks/llm/graph_tokenizer.py` | **Create** | E_c → G_LLM via learnable queries + MHA |
| `mtgs/networks/llm/memory_attn.py` | **Create** | MemoryCrossAttn (h + gate·CrossAttn(h,G)) + layer wrapper |
| `mtgs/networks/llm/reasoner.py` | **Create** | EvidenceAugmentedLLM: load Qwen3.5-4B, wrap Gated Attention layers, grounding, Yes/No loss |
| `mtgs/datasets/gaze_qa.py` | **Create** | GazeQACollator: batch → list of (task,i,j,label) tuples |
| `mtgs/networks/llm/llm_trainer.py` | **Create** | LLMReasonerModel (pl.LightningModule) |
| `mtgs/datasets/llm_datamodule.py` | **Create** | LLMDataModule: same as VSGaze but val_transform for train |
| `mtgs/config/config.yaml` | **Modify** | append `interaction.llm` block (additive only) |
| `scripts/main_llm.py` | **Create** | Stage B Hydra entry point |
| `scripts/train_llm_align.sh` | **Create** | SLURM script for Stage B |
| `tests/test_graph_tokenizer.py` | **Create** | unit test for tokenizer |
| `tests/test_memory_attn.py` | **Create** | unit test for cross-attn + layer wrapper |
| `tests/test_gaze_qa.py` | **Create** | unit test for QA collator |
| `tests/test_llm_trainer_hook.py` | **Create** | smoke test: hook captures correct shapes |

---

## Task 1: GraphEvidenceTokenizer

**Files:**
- Create: `mtgs/networks/llm/__init__.py`
- Create: `mtgs/networks/llm/graph_tokenizer.py`
- Create: `tests/test_graph_tokenizer.py`

- [ ] **Step 1.1 — Write failing test**

```python
# tests/test_graph_tokenizer.py
import torch
import pytest
from mtgs.networks.llm.graph_tokenizer import GraphEvidenceTokenizer

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
```

- [ ] **Step 1.2 — Run test to confirm it fails**

```bash
cd /home/jinwoongjung/MTGS
conda run -n mtgs python -m pytest tests/test_graph_tokenizer.py -v
```
Expected: `ModuleNotFoundError: No module named 'mtgs.networks.llm'`

- [ ] **Step 1.3 — Create package init**

```python
# mtgs/networks/llm/__init__.py
```
(empty file)

- [ ] **Step 1.4 — Implement GraphEvidenceTokenizer**

```python
# mtgs/networks/llm/graph_tokenizer.py
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
```

- [ ] **Step 1.5 — Run tests to confirm they pass**

```bash
conda run -n mtgs python -m pytest tests/test_graph_tokenizer.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 1.6 — Commit**

```bash
git add mtgs/networks/llm/__init__.py mtgs/networks/llm/graph_tokenizer.py tests/test_graph_tokenizer.py
git commit -m "feat: add GraphEvidenceTokenizer (Stage 4)"
```

---

## Task 2: MemoryCrossAttn + MemoryAugmentedLayer

**Files:**
- Create: `mtgs/networks/llm/memory_attn.py`
- Create: `tests/test_memory_attn.py`

- [ ] **Step 2.1 — Write failing test**

```python
# tests/test_memory_attn.py
import torch
import pytest
from mtgs.networks.llm.memory_attn import MemoryCrossAttn, MemoryAugmentedLayer


def test_gate_zero_init_is_identity():
    """With gate=0, output must equal input exactly."""
    ca = MemoryCrossAttn(d_model=32, num_heads=4)
    assert ca.gate.item() == 0.0
    B, S, M = 2, 10, 8
    h = torch.randn(B, S, 32)
    G = torch.randn(B, M, 32)
    out = ca(h, G)
    assert torch.allclose(out, h), "gate=0 should be identity"


def test_cross_attn_output_shape():
    ca = MemoryCrossAttn(d_model=64, num_heads=4)
    h = torch.randn(3, 15, 64)
    G = torch.randn(3, 32, 64)
    out = ca(h, G)
    assert out.shape == h.shape


class _DummyLayer(torch.nn.Module):
    """Simulates a transformer layer that returns (hidden_states, *extras)."""
    def forward(self, h, **kwargs):
        return (h * 2.0, torch.tensor(1.0))  # two-element output


def test_memory_augmented_layer_no_G():
    """Without G_LLM set, wrapper is transparent (passes through to original)."""
    orig = _DummyLayer()
    ca = MemoryCrossAttn(d_model=8, num_heads=2)
    layer = MemoryAugmentedLayer(orig, ca)
    h = torch.randn(1, 5, 8)
    out = layer(h)
    assert torch.allclose(out[0], h * 2.0)


def test_memory_augmented_layer_with_G():
    """With G_LLM set and gate trained, output differs from input*2."""
    orig = _DummyLayer()
    ca = MemoryCrossAttn(d_model=8, num_heads=2)
    # Force gate != 0
    with torch.no_grad():
        ca.gate.fill_(1.0)
    layer = MemoryAugmentedLayer(orig, ca)
    h = torch.randn(1, 5, 8)
    G = torch.randn(1, 4, 8)
    layer._G_LLM = G
    out = layer(h)
    # Should NOT equal h*2.0 because cross-attn adds something
    assert not torch.allclose(out[0], h * 2.0)
```

- [ ] **Step 2.2 — Run test to confirm failure**

```bash
conda run -n mtgs python -m pytest tests/test_memory_attn.py -v
```
Expected: `ImportError`

- [ ] **Step 2.3 — Implement**

```python
# mtgs/networks/llm/memory_attn.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MemoryCrossAttn(nn.Module):
    """Cross-attention from LLM hidden states to G_LLM evidence tokens.

    h_out = h + gate * LN(CrossAttn(q=h, kv=G_LLM))
    gate is initialized to 0 so the layer starts as identity.
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: Tensor, G: Tensor) -> Tensor:
        # h: (B, seq_len, d_model)
        # G: (B, M, d_model)
        delta, _ = self.attn(h, G, G)
        return h + self.gate * self.norm(delta)


class MemoryAugmentedLayer(nn.Module):
    """Wraps one transformer layer, injecting cross-attn when _G_LLM is set.

    Usage:
        layer._G_LLM = G_LLM   # before forward
        output = layer(...)
        layer._G_LLM = None    # after forward
    """

    def __init__(self, original_layer: nn.Module, cross_attn: MemoryCrossAttn):
        super().__init__()
        self.layer = original_layer
        self.cross_attn = cross_attn
        self._G_LLM: Optional[Tensor] = None

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if self._G_LLM is not None:
            h = out[0]                              # hidden states are first element
            h = self.cross_attn(h, self._G_LLM)
            out = (h,) + out[1:]
        return out
```

- [ ] **Step 2.4 — Run tests**

```bash
conda run -n mtgs python -m pytest tests/test_memory_attn.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 2.5 — Commit**

```bash
git add mtgs/networks/llm/memory_attn.py tests/test_memory_attn.py
git commit -m "feat: add MemoryCrossAttn and MemoryAugmentedLayer (Stage 5 building blocks)"
```

---

## Task 3: GazeQACollator

**Files:**
- Create: `mtgs/datasets/gaze_qa.py`
- Create: `tests/test_gaze_qa.py`

> **VLM update:** `QAPair` now carries `src_bbox` and `dst_bbox` (normalized [x1,y1,x2,y2] tuples)
> extracted from `head_bboxes[b, t_c, idx]`. These are used by Task 4 to build bbox-in-text prompts.

- [ ] **Step 3.1 — Write failing test**

```python
# tests/test_gaze_qa.py
import torch
import itertools
import pytest
from mtgs.datasets.gaze_qa import GazeQACollator, QAPair


def _make_batch(B=2, T=5, N=4):
    """Synthetic batch with the same key structure as pad_collate_fn output."""
    P = N * (N - 1)
    batch = {
        "lah_labels":   torch.randint(-1, 2, (B, T, P)).float(),
        "laeo_labels":  torch.full((B, T, P), -1.0),  # no LAEO annotation
        "coatt_labels": torch.randint(-1, 2, (B, T, P)).float(),
        "num_valid_people": torch.full((B, 1), N, dtype=torch.long),
        "head_bboxes":  torch.rand(B, T, N, 4),  # normalized [x1,y1,x2,y2]
    }
    # Force at least one valid label per batch item
    batch["lah_labels"][:, T // 2, 0] = 1.0
    batch["lah_labels"][:, T // 2, 1] = 0.0
    return batch


def test_returns_qa_pairs():
    collator = GazeQACollator()
    batch = _make_batch(B=2, N=4)
    pairs = collator(batch)
    assert len(pairs) > 0
    for p in pairs:
        assert isinstance(p, QAPair)
        assert p.task in ("lah", "laeo", "sa")
        assert p.label in (0, 1)
        assert p.batch_idx < 2
        assert len(p.src_bbox) == 4
        assert len(p.dst_bbox) == 4


def test_skips_minus_one_labels():
    collator = GazeQACollator()
    batch = _make_batch(B=1, N=4)
    batch["lah_labels"][:] = -1.0
    batch["coatt_labels"][:] = -1.0
    pairs = collator(batch)
    assert len(pairs) == 0


def test_laeo_skipped_when_all_minus_one():
    collator = GazeQACollator()
    batch = _make_batch(B=1, N=4)
    # laeo is all -1 → no laeo pairs
    laeo_pairs = [p for p in collator(batch) if p.task == "laeo"]
    assert len(laeo_pairs) == 0


def test_balanced_sampling():
    """With many valid pairs, balanced collator returns 1:1 pos:neg per task."""
    collator = GazeQACollator(balanced=True, max_pairs_per_task=4)
    B, T, N = 1, 5, 6
    P = N * (N - 1)
    batch = {
        "lah_labels":   torch.zeros(B, T, P),
        "laeo_labels":  torch.full((B, T, P), -1.0),
        "coatt_labels": torch.full((B, T, P), -1.0),
        "num_valid_people": torch.full((B, 1), N, dtype=torch.long),
        "head_bboxes":  torch.rand(B, T, N, 4),
    }
    # Force some positives
    batch["lah_labels"][0, T // 2, :3] = 1.0
    pairs = collator(batch)
    lah_pairs = [p for p in pairs if p.task == "lah"]
    pos = sum(p.label == 1 for p in lah_pairs)
    neg = sum(p.label == 0 for p in lah_pairs)
    assert pos == neg, f"Expected balanced, got pos={pos} neg={neg}"


def test_bbox_values_match_head_bboxes():
    """src_bbox/dst_bbox must match head_bboxes[b, t_c, idx]."""
    collator = GazeQACollator(balanced=False)
    batch = _make_batch(B=1, N=4)
    T = batch["lah_labels"].shape[1]
    t_c = T // 2
    pairs = collator(batch)
    for p in pairs:
        expected_src = tuple(batch["head_bboxes"][p.batch_idx, t_c, p.src_idx].tolist())
        expected_dst = tuple(batch["head_bboxes"][p.batch_idx, t_c, p.dst_idx].tolist())
        assert p.src_bbox == pytest.approx(expected_src, abs=1e-5)
        assert p.dst_bbox == pytest.approx(expected_dst, abs=1e-5)
```

- [ ] **Step 3.2 — Run test to confirm failure**

```bash
cd /home/jinwoongjung/MTGS && conda run -n mtgs python -m pytest tests/test_gaze_qa.py -v
```

- [ ] **Step 3.3 — Implement**

```python
# mtgs/datasets/gaze_qa.py
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class QAPair:
    batch_idx: int           # which item in the batch
    task: str                # "lah" | "laeo" | "sa"
    src_idx: int             # looker (LAH subject) or first person (LAEO/SA)
    dst_idx: int             # target (LAH object) or second person (LAEO/SA)
    label: int               # 1 = Yes, 0 = No
    src_bbox: Tuple[float, float, float, float] = field(default=(0., 0., 1., 1.))
    dst_bbox: Tuple[float, float, float, float] = field(default=(0., 0., 1., 1.))


class GazeQACollator:
    """Generates Yes/No QA pairs from a padded batch at the center frame.

    Pair convention (from mtgs_net.py):
        pairs = list(itertools.permutations(range(N), 2))
        pairs[k] = (src_k, dst_k)
        lah_labels[b, t, k] = 1  means  person src_k looks at person dst_k
        → LAH query: "Does <P> [src_bbox] look at <P> [dst_bbox]?"

    Valid persons are right-aligned: indices [N - nv, N).
    Bboxes extracted from head_bboxes[b, t_c, idx] (normalized [x1,y1,x2,y2]).
    """

    def __init__(self, balanced: bool = True, max_pairs_per_task: int = 32):
        self.balanced = balanced
        self.max_pairs_per_task = max_pairs_per_task

    def __call__(self, batch: dict) -> List[QAPair]:
        B = batch["lah_labels"].shape[0]
        T = batch["lah_labels"].shape[1]
        t_c = T // 2
        P = batch["lah_labels"].shape[2]
        N_padded = int(round((1 + math.sqrt(1 + 4 * P)) / 2))

        pairs_idx = list(itertools.permutations(range(N_padded), 2))
        head_bboxes = batch.get("head_bboxes")  # (B, T, N, 4) or None

        all_pairs: List[QAPair] = []
        for b in range(B):
            nv = int(batch["num_valid_people"][b, 0].item())
            valid_start = N_padded - nv

            for task, label_key in [("lah", "lah_labels"),
                                     ("laeo", "laeo_labels"),
                                     ("sa", "coatt_labels")]:
                labels = batch[label_key][b, t_c]  # (P,)
                pos, neg = [], []
                for k, (src_k, dst_k) in enumerate(pairs_idx):
                    if src_k < valid_start or dst_k < valid_start:
                        continue
                    lbl = int(labels[k].item())
                    if lbl == -1:
                        continue
                    src_bbox = tuple(head_bboxes[b, t_c, src_k].tolist()) \
                        if head_bboxes is not None else (0., 0., 1., 1.)
                    dst_bbox = tuple(head_bboxes[b, t_c, dst_k].tolist()) \
                        if head_bboxes is not None else (0., 0., 1., 1.)
                    qa = QAPair(b, task, src_k, dst_k, lbl, src_bbox, dst_bbox)
                    (pos if lbl == 1 else neg).append(qa)

                if self.balanced and pos and neg:
                    n_each = min(len(pos), len(neg), self.max_pairs_per_task // 2)
                    all_pairs += random.sample(pos, n_each)
                    all_pairs += random.sample(neg, n_each)
                else:
                    all_pairs += pos + neg

        return all_pairs
```

- [ ] **Step 3.4 — Run tests**

```bash
cd /home/jinwoongjung/MTGS && conda run -n mtgs python -m pytest tests/test_gaze_qa.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 3.5 — Commit**

```bash
git add mtgs/datasets/gaze_qa.py tests/test_gaze_qa.py
git commit -m "feat: add GazeQACollator with bbox grounding fields in QAPair"
```

---

## Task 4: EvidenceAugmentedLLM

**Files:**
- Create: `mtgs/networks/llm/reasoner.py`

> **VLM update:** Backbone is now `Qwen/Qwen3-VL-8B-Instruct`. Entity grounding combines:
> (1) `<P>` embedding + W_node·v (semantic graph features), AND
> (2) bbox text `[x1,y1,x2,y2]` appended after `<P>` in the prompt (spatial grounding).
>
> No unit test — loading the 8B model requires VRAM. Integration test is in Task 5.

- [ ] **Step 4.1 — Implement EvidenceAugmentedLLM**

```python
# mtgs/networks/llm/reasoner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

from mtgs.networks.llm.graph_tokenizer import GraphEvidenceTokenizer
from mtgs.networks.llm.memory_attn import MemoryCrossAttn, MemoryAugmentedLayer
from mtgs.datasets.gaze_qa import QAPair


# Qwen3-VL-8B (Qwen3-8B backbone): 36 layers, full_attention_interval=4
# GatedAttention (full attn) at every 4th layer starting from index 3
_DEFAULT_CROSS_ATTN_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35]


def _fmt_bbox(bbox: Tuple[float, float, float, float]) -> str:
    return "[{:.2f},{:.2f},{:.2f},{:.2f}]".format(*bbox)


class EvidenceAugmentedLLM(nn.Module):
    """Frozen Qwen3-VL-8B augmented with graph evidence via cross-attention.

    Entity grounding:
      - Text level: bbox [x1,y1,x2,y2] appended after each <P> token in prompt
      - Embedding level: Emb(<P>) += W_node · v  (v = v_src or v_tgt)

    Trainable: MemoryCrossAttn layers (9), W_node, <P> token embedding, Q_g, W_proj.
    Frozen: all other VLM parameters and MTGS pipeline.
    """

    def __init__(self, cfg):
        super().__init__()
        llm_cfg = cfg.interaction.llm
        edge_dim = cfg.interaction.gaze_graph.edge_dim

        # ── Load tokenizer + VLM (text-only path) ─────────────────────────────
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            llm_cfg.backbone, trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_cfg.backbone,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in self.llm.parameters():
            p.requires_grad_(False)

        d_llm = self.llm.config.hidden_size  # e.g. 3584 for Qwen3-VL-8B

        # ── Special <P> token ─────────────────────────────────────────────────
        self.hf_tokenizer.add_special_tokens({"additional_special_tokens": ["<P>"]})
        self.llm.resize_token_embeddings(len(self.hf_tokenizer))
        self._P_token_id = self.hf_tokenizer.convert_tokens_to_ids("<P>")
        self._yes_id = self.hf_tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id  = self.hf_tokenizer.encode("No",  add_special_tokens=False)[0]

        # ── Entity grounding: edge_dim → d_llm ───────────────────────────────
        self.W_node = nn.Linear(edge_dim, d_llm, bias=False)

        # ── Graph tokenizer (Stage 4) ─────────────────────────────────────────
        self.graph_tokenizer = GraphEvidenceTokenizer(
            edge_dim=edge_dim,
            d_llm=d_llm,
            m=llm_cfg.memory_tokens_m,
            depth=llm_cfg.tokenizer_depth,
        )

        # ── Wrap Full-Attention layers with MemoryCrossAttn ───────────────────
        indices = list(llm_cfg.get("cross_attn_layer_indices",
                                   _DEFAULT_CROSS_ATTN_INDICES))
        self._cross_attn_indices = indices
        for idx in indices:
            orig = self.llm.model.layers[idx]
            cross_attn = MemoryCrossAttn(d_llm, num_heads=8)
            self.llm.model.layers[idx] = MemoryAugmentedLayer(orig, cross_attn)

    # ── G_LLM injection helpers ───────────────────────────────────────────────

    def _set_G(self, G: Tensor):
        for idx in self._cross_attn_indices:
            self.llm.model.layers[idx]._G_LLM = G

    def _clear_G(self):
        for idx in self._cross_attn_indices:
            self.llm.model.layers[idx]._G_LLM = None

    # ── Query construction with bbox text + node feature grounding ────────────

    def _build_input_embeds(
        self,
        qa: QAPair,
        v_src_c: Tensor,   # (B, N, De)
        v_tgt_c: Tensor,   # (B, Tl, De)  Tl = N+2
        device: torch.device,
    ) -> Tensor:
        """Tokenize the bbox-augmented prompt and apply W_node grounding to <P> positions.

        Prompt format examples:
          LAH:  "Does <P> [0.10,0.20,0.30,0.40] look at <P> [0.50,0.60,0.70,0.80]? Answer:"
          LAEO: "Do <P> [..] and <P> [..] look at each other? Answer:"
          SA:   "Do <P> [..] and <P> [..] attend to the same target? Answer:"

        <P> embedding is further grounded: Emb(<P>) += W_node(v)
          subject: v_src[src_idx]
          object:  v_tgt[dst_idx]  ← LAH
                   v_src[dst_idx]  ← LAEO/SA

        Returns: (1, seq_len, d_llm)
        """
        src_bbox_str = _fmt_bbox(qa.src_bbox)
        dst_bbox_str = _fmt_bbox(qa.dst_bbox)

        templates = {
            "lah":  f"Does <P> {src_bbox_str} look at <P> {dst_bbox_str}? Answer:",
            "laeo": f"Do <P> {src_bbox_str} and <P> {dst_bbox_str} look at each other? Answer:",
            "sa":   f"Do <P> {src_bbox_str} and <P> {dst_bbox_str} attend to the same target? Answer:",
        }
        prompt = templates[qa.task]
        input_ids = self.hf_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        emb_table = self.llm.get_input_embeddings()
        embeds = emb_table(input_ids).float()  # (1, seq_len, d_llm)

        P_id = self._P_token_id
        p_positions = (input_ids[0] == P_id).nonzero(as_tuple=True)[0]

        b = qa.batch_idx
        v_subj = self.W_node(v_src_c[b, qa.src_idx].float())
        if qa.task == "lah":
            v_obj = self.W_node(v_tgt_c[b, qa.dst_idx].float())
        else:
            v_obj = self.W_node(v_src_c[b, qa.dst_idx].float())

        if len(p_positions) >= 1:
            embeds[0, p_positions[0]] = embeds[0, p_positions[0]] + v_subj
        if len(p_positions) >= 2:
            embeds[0, p_positions[1]] = embeds[0, p_positions[1]] + v_obj

        return embeds

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        E_c: Tensor,           # (B, N, Tl, De)
        edge_valid: Tensor,    # (B, N, 2N+2)
        v_src_c: Tensor,       # (B, N, De)
        v_tgt_c: Tensor,       # (B, Tl, De)
        qa_pairs: List[QAPair],
    ) -> Tensor:
        if not qa_pairs:
            return torch.tensor(0.0, requires_grad=True,
                                device=E_c.device, dtype=torch.float32)

        device = E_c.device
        G_LLM = self.graph_tokenizer(E_c, edge_valid)       # (B, M, d_llm)
        G_LLM_bf16 = G_LLM.to(torch.bfloat16)

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for qa in qa_pairs:
            self._set_G(G_LLM_bf16[qa.batch_idx : qa.batch_idx + 1])
            embeds = self._build_input_embeds(qa, v_src_c, v_tgt_c, device)
            embeds_bf16 = embeds.to(torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self.llm(inputs_embeds=embeds_bf16)

            logits = out.logits[0, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_id = self._yes_id if qa.label == 1 else self._no_id
            loss = -log_probs[target_id]
            total_loss = total_loss + loss
            count += 1

        self._clear_G()
        return total_loss / count
```

- [ ] **Step 4.2 — Commit**

```bash
git add mtgs/networks/llm/reasoner.py
git commit -m "feat: add EvidenceAugmentedLLM with Qwen3-VL-8B, bbox-in-text grounding, MemoryCrossAttn injection"
```

---

## Task 5: LLMReasonerModel + hook smoke test

**Files:**
- Create: `mtgs/networks/llm/llm_trainer.py`
- Create: `tests/test_llm_trainer_hook.py`

- [ ] **Step 5.1 — Write hook smoke test (CPU, no LLM)**

```python
# tests/test_llm_trainer_hook.py
"""Verifies that the _UnifiedRefiner forward hook captures correct shapes.
Does NOT load Qwen3.5-4B. Tests only the hook mechanism."""
import torch
import pytest
from mtgs.networks.adaptor_modules import GazeGraphBlock

# Minimal config mock
class _Cfg:
    class interaction:
        class gaze_graph:
            edge_dim = 16
            num_layers = 1
            use_prior = False
            prior_weight = 0.5
            use_node_xattn = False

def test_refiner_hook_captures_shapes():
    B, T, N, D = 1, 5, 3, 64  # small for CPU test
    cfg_gg = _Cfg.interaction.gaze_graph
    block = GazeGraphBlock(
        token_dim=D,
        edge_dim=cfg_gg.edge_dim,
        num_layers=cfg_gg.num_layers,
        use_prior=cfg_gg.use_prior,
        use_node_xattn=cfg_gg.use_node_xattn,
    )

    captured = {}

    def hook(module, inp, output):
        E, v_src, v_tgt = output
        captured["E"] = E
        captured["v_src"] = v_src
        captured["v_tgt"] = v_tgt

    block.refiner.register_forward_hook(hook)

    # Minimal forward pass inputs
    person_tokens = torch.randn(B, T, N, D)
    num_valid = torch.full((B,), N, dtype=torch.long)
    gaze_vecs = torch.randn(B, T, N, 2)
    head_bboxes = torch.rand(B, T, N, 4)
    gaze_hm = torch.rand(B, T, N, 8, 8)
    inout = torch.zeros(B, T, N)

    block(person_tokens, num_valid, gaze_vecs, head_bboxes, gaze_hm, inout)

    De = cfg_gg.edge_dim
    Tl = N + 2
    assert captured["E"].shape == (B, T, N, Tl, De), f"E shape mismatch: {captured['E'].shape}"
    assert captured["v_src"].shape == (B, T, N, De)
    assert captured["v_tgt"].shape == (B, T, Tl, De)
```

- [ ] **Step 5.2 — Run hook test**

```bash
conda run -n mtgs python -m pytest tests/test_llm_trainer_hook.py -v
```
Expected: PASS (no LLM loaded, just verifies hook captures correct shapes)

- [ ] **Step 5.3 — Implement LLMReasonerModel**

```python
# mtgs/networks/llm/llm_trainer.py
import torch
import torch.nn as nn
import lightning.pytorch as pl
from omegaconf import DictConfig

from mtgs.networks import MTGS
from mtgs.networks.llm.reasoner import EvidenceAugmentedLLM
from mtgs.datasets.gaze_qa import GazeQACollator


class LLMReasonerModel(pl.LightningModule):
    """Stage B: trains EvidenceAugmentedLLM on top of frozen gaze_graph.

    Hooks into _UnifiedRefiner to capture E, v_src, v_tgt without
    modifying any existing code.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._graph_states: dict = {}

        # ── Frozen MTGS (Stage A checkpoint loaded in configure_model) ────────
        self.frozen_mtgs: MTGS = MTGS(
            encoder_name=cfg.encoder.name,
            interaction_type="gaze_graph",
            gaze_graph_edge_dim=cfg.interaction.gaze_graph.edge_dim,
            gaze_graph_num_layers=cfg.interaction.gaze_graph.num_layers,
            gaze_graph_use_prior=cfg.interaction.gaze_graph.use_prior,
            gaze_graph_prior_weight=cfg.interaction.gaze_graph.prior_weight,
            gaze_graph_use_node_xattn=cfg.interaction.gaze_graph.use_node_xattn,
        )
        for p in self.frozen_mtgs.parameters():
            p.requires_grad_(False)
        self.frozen_mtgs.eval()

        # ── Hook: capture refiner output (E, v_src, v_tgt) ───────────────────
        def _refiner_hook(module, inp, output):
            E, v_src, v_tgt = output
            self._graph_states["E"] = E.detach()
            self._graph_states["v_src"] = v_src.detach()
            self._graph_states["v_tgt"] = v_tgt.detach()

        def _block_hook(module, inp, output):
            # output = (lah, laeo, sa, null_in, null_out, edge_valid)
            self._graph_states["edge_valid"] = output[5].detach()

        self.frozen_mtgs.gaze_graph_block.refiner.register_forward_hook(_refiner_hook)
        self.frozen_mtgs.gaze_graph_block.register_forward_hook(_block_hook)

        # ── Trainable LLM module ──────────────────────────────────────────────
        self.llm_model = EvidenceAugmentedLLM(cfg)

        # ── QA pair generator ─────────────────────────────────────────────────
        self.qa_collator = GazeQACollator(
            balanced=True,
            max_pairs_per_task=cfg.interaction.llm.get("max_pairs_per_task", 32),
        )

    def load_stage_a_weights(self, ckpt_path: str):
        """Load Stage A gaze_graph checkpoint into frozen_mtgs."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # MTGSModel wraps MTGS as self.model — strip the prefix
        state = {k.replace("model.", "", 1): v
                 for k, v in ckpt["state_dict"].items()
                 if k.startswith("model.")}
        missing, unexpected = self.frozen_mtgs.load_state_dict(state, strict=False)
        if missing:
            print(f"[LLMReasonerModel] Missing keys: {missing[:5]}...")
        self.frozen_mtgs.eval()
        for p in self.frozen_mtgs.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        # 1. Run frozen MTGS to trigger hooks (val_transform → deterministic)
        with torch.no_grad():
            self.frozen_mtgs.eval()
            self.frozen_mtgs(batch)

        E = self._graph_states["E"]            # (B, T, N, Tl, De)
        v_src = self._graph_states["v_src"]    # (B, T, N, De)
        v_tgt = self._graph_states["v_tgt"]    # (B, T, Tl, De)
        edge_valid = self._graph_states["edge_valid"]  # (B, N, 2N+2)

        t_c = E.shape[1] // 2
        E_c = E[:, t_c]           # (B, N, Tl, De)
        v_src_c = v_src[:, t_c]   # (B, N, De)
        v_tgt_c = v_tgt[:, t_c]   # (B, Tl, De)

        # 2. Generate QA pairs from batch labels at center frame
        qa_pairs = self.qa_collator(batch)
        if not qa_pairs:
            return None

        # 3. LLM forward → Yes/No loss
        loss = self.llm_model(E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs)
        self.log("train/loss_llm", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.frozen_mtgs.eval()
            self.frozen_mtgs(batch)
            E = self._graph_states["E"]
            v_src = self._graph_states["v_src"]
            v_tgt = self._graph_states["v_tgt"]
            edge_valid = self._graph_states["edge_valid"]
            t_c = E.shape[1] // 2
            E_c, v_src_c, v_tgt_c = E[:, t_c], v_src[:, t_c], v_tgt[:, t_c]
            qa_pairs = self.qa_collator(batch)
            if not qa_pairs:
                return
            loss = self.llm_model(E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs)
        self.log("val/loss_llm", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        params = [p for p in self.llm_model.parameters() if p.requires_grad]
        lr = self.cfg.interaction.llm.get("lr", 1e-4)
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
```

- [ ] **Step 5.4 — Run hook test again (confirms model can be instantiated without LLM)**

```bash
conda run -n mtgs python -m pytest tests/test_llm_trainer_hook.py -v
```
Expected: PASS (unchanged)

- [ ] **Step 5.5 — Commit**

```bash
git add mtgs/networks/llm/llm_trainer.py tests/test_llm_trainer_hook.py
git commit -m "feat: add LLMReasonerModel with refiner hook (Stage B Lightning module)"
```

---

## Task 6: LLMDataModule (val_transform for train)

**Files:**
- Create: `mtgs/datasets/llm_datamodule.py`

> C15 confirmed: Stage B uses val_transform (no RandomCrop/ColorJitter) for train split
> so frozen graph produces deterministic E.

- [ ] **Step 6.1 — Implement**

```python
# mtgs/datasets/llm_datamodule.py
"""Stage B datamodule: identical to VSGaze but forces val_transform on train split."""
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset

from mtgs.datasets.videoattentiontarget_temporal import VideoAttentionTargetDataset_temporal
from mtgs.datasets.childplay_temporal import ChildPlayDataset_temporal
from mtgs.datasets.uco_laeo_temporal import VideoLAEODataset_temporal
from mtgs.datasets.videocoatt_temporal import VideoCoAttDataset_temporal
from mtgs.train.transforms import Resize, ToTensor, Normalize, Compose
from mtgs.train.collate import pad_collate_fn
from mtgs.datasets.vsgaze import IMG_MEAN, IMG_STD


class LLMDataModule(pl.LightningDataModule):
    """VSGaze datasets with val_transform for all splits (no random augmentation)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _make_transform(self):
        return Compose([
            Resize(img_size=self.cfg.data.image_size,
                   head_size=self.cfg.data.head_size),
            ToTensor(),
            Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
        ])

    def _make_dataset(self, split):
        cfg = self.cfg
        t = self._make_transform()
        stride = max(3, cfg.data.temporal_context * cfg.data.temporal_stride * 2)
        kw = dict(
            split=split,
            stride=stride,
            transform=t,
            tr=(-0.1, 0.1) if split == "train" else (0.0, 0.0),
            num_people=cfg.data.num_people[split] if split != "train"
                       else cfg.data.num_people.get("train", "all"),
            temporal_context=cfg.data.temporal_context,
            temporal_stride=cfg.data.temporal_stride,
            image_size=cfg.data.image_size,
        )
        datasets = []
        if cfg.data.get("root_vat"):
            datasets.append(VideoAttentionTargetDataset_temporal(
                root=cfg.data.root_vat, ann_root=cfg.data.ann_root, **kw))
        if cfg.data.get("root_childplay"):
            datasets.append(ChildPlayDataset_temporal(
                root=cfg.data.root_childplay, ann_root=cfg.data.ann_root, **kw))
        if cfg.data.get("root_laeo"):
            datasets.append(VideoLAEODataset_temporal(
                root=cfg.data.root_laeo, ann_root=cfg.data.ann_root, **kw))
        if cfg.data.get("root_coatt"):
            datasets.append(VideoCoAttDataset_temporal(
                root=cfg.data.root_coatt, ann_root=cfg.data.ann_root, **kw))
        return ConcatDataset(datasets)

    def setup(self, stage=None):
        self.train_ds = self._make_dataset("train")
        self.val_ds   = self._make_dataset("val")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=True,
        )
```

- [ ] **Step 6.2 — Commit**

```bash
git add mtgs/datasets/llm_datamodule.py
git commit -m "feat: add LLMDataModule with val_transform for Stage B (no augmentation)"
```

---

## Task 7: Config block + entry point + SLURM script

**Files:**
- Modify: `mtgs/config/config.yaml` (append only)
- Create: `scripts/main_llm.py`
- Create: `scripts/train_llm_align.sh`

- [ ] **Step 7.1 — Append `interaction.llm` block to config.yaml**

Open `mtgs/config/config.yaml` and append the following **at the end of the `interaction:` block**
(after the existing `gaze_graph:` subsection, before the next top-level key):

```yaml
  # ── VLM Reasoner (Stage B) ─────────────────────────────────────────────────
  llm:
    backbone: "Qwen/Qwen3-VL-8B-Instruct"    # HF repo or local path
    load_dtype: "bf16"
    memory_tokens_m: 32                       # G_LLM token count M
    tokenizer_depth: 1                        # set-to-token MHA depth
    # Qwen3-VL-8B: 36 layers, full_attention_interval=4 → Full Attention at every 4th
    cross_attn_layer_indices: [3,7,11,15,19,23,27,31,35]
    gate_init: 0.0                            # g_ℓ 0-init (identity at start)
    tasks: ["lah", "laeo", "sa"]
    max_pairs_per_task: 32                    # balanced sampling cap per clip
    lr: 1e-4
    use_val_transform: true                   # C15: no augmentation during Stage B
```

- [ ] **Step 7.2 — Verify existing pipeline is unaffected**

```bash
# Dry-run: parse config with hydra, check no existing key changed
conda run -n mtgs python -c "
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir='/home/jinwoongjung/MTGS/mtgs/config', version_base=None):
    cfg = compose(config_name='config.yaml')
    assert cfg.interaction.type == 'gaze_graph'
    assert cfg.interaction.gaze_graph.edge_dim == 128
    assert hasattr(cfg.interaction, 'llm')
    print('Config OK, llm block:', cfg.interaction.llm.backbone)
"
```
Expected output: `Config OK, llm block: Qwen/Qwen3-VL-8B-Instruct`

- [ ] **Step 7.3 — Create main_llm.py**

```python
# scripts/main_llm.py
"""Stage B entry point — separate from main.py to avoid any interference."""
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import torch

from mtgs.config import ConfigManager
from mtgs.networks.llm.llm_trainer import LLMReasonerModel
from mtgs.datasets.llm_datamodule import LLMDataModule


@hydra.main(
    config_path="./../mtgs/config/",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)
    pl.seed_everything(cfg.train.get("seed", 42))
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    model = LLMReasonerModel(cfg)

    # Load Stage A checkpoint
    stage_a_ckpt = cfg.interaction.llm.get("stage_a_ckpt", None)
    if stage_a_ckpt:
        model.load_stage_a_weights(stage_a_ckpt)
    else:
        raise ValueError("interaction.llm.stage_a_ckpt must be set for Stage B training")

    datamodule = LLMDataModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=50,
        val_check_interval=1.0,
        default_root_dir=cfg.experiment.output_folder,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.4 — Create SLURM script**

```bash
# scripts/train_llm_align.sh
#!/bin/bash
#SBATCH --job-name=llm_align
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/llm_align_%j.out
#SBATCH --error=logs/llm_align_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

STAGE_A_CKPT="/path/to/stage_a_gaze_graph.ckpt"  # ← set before submitting
EXP_NAME="llm_align_v1"

python main_llm.py \
    experiment.name="${EXP_NAME}" \
    experiment.output_folder="../experiments/${EXP_NAME}" \
    interaction.type=gaze_graph \
    interaction.llm.stage_a_ckpt="${STAGE_A_CKPT}" \
    train.epochs=20 \
    train.batch_size=1 \
    train.accumulate_grad_batches=4 \
    train.num_workers=4 \
    data.num_people.train=11
```

- [ ] **Step 7.5 — Run all tests together to confirm nothing is broken**

```bash
conda run -n mtgs python -m pytest tests/ -v
```
Expected: all tests PASS (graph_tokenizer × 3, memory_attn × 4, gaze_qa × 4, hook × 1 = 12 tests)

- [ ] **Step 7.6 — Commit**

```bash
git add mtgs/config/config.yaml scripts/main_llm.py scripts/train_llm_align.sh
git commit -m "feat: add Stage B config block, entry point, and SLURM script"
```

---

## Self-Review

### Spec coverage check

| LLM_TODO.md item | Covered by task |
|-----------------|----------------|
| `0.1` GazeGraphBlock state exposure | ✅ Task 5 (forward hook on refiner — no code change needed) |
| `1` GraphEvidenceTokenizer | ✅ Task 1 |
| `2` EvidenceAugmentedLLM | ✅ Task 4 |
| `2.1` Entity grounding (v_src/v_tgt, W_node) | ✅ Task 4 (`_build_input_embeds`) |
| `2.2` MemoryCrossAttn + gate 0-init | ✅ Task 2 |
| `2.3` Yes/No CE loss | ✅ Task 4 (`forward`) |
| `2.5` All pairs → LLM (no routing) | ✅ Task 3 (GazeQACollator) + Task 4 |
| `3` QA generation (LAH/LAEO/SA, label availability) | ✅ Task 3 |
| `4` Stage B freeze + val_transform | ✅ Task 5 (frozen_mtgs) + Task 6 (LLMDataModule) |
| C6: Gated Attention indices [3,7,11,15,19,23,27,31] | ✅ Task 4 (`_DEFAULT_CROSS_ATTN_INDICES`) |
| C7: single `<P>` token + W_node grounding | ✅ Task 4 |
| C12: balanced 1:1 sampling | ✅ Task 3 (`GazeQACollator(balanced=True)`) |
| C15: val_transform for Stage B | ✅ Task 6 |
| config `interaction.llm` block | ✅ Task 7 |

### Existing pipeline non-interference check

| Existing file | Modified? | Reason safe |
|---------------|-----------|-------------|
| `adaptor_modules.py` | ❌ NOT touched | hook registered externally |
| `mtgs_net.py` | ❌ NOT touched | hook registered externally |
| `models.py` | ❌ NOT touched | separate Lightning module |
| `vsgaze.py` | ❌ NOT touched | new LLMDataModule |
| `main.py` | ❌ NOT touched | new `main_llm.py` |
| `config.yaml` | ✅ additive only | new `interaction.llm` block appended |

### Placeholder scan
- No "TBD" or "TODO" in implementation steps ✓
- All shapes annotated with concrete values ✓
- `STAGE_A_CKPT` in shell script intentionally left as placeholder — user must fill before submitting ✓

### Type consistency check
- `GraphEvidenceTokenizer.forward(E_c, edge_valid)` — matches usage in `EvidenceAugmentedLLM.forward` ✓
- `GazeQACollator.__call__(batch) → List[QAPair]` — matches `LLMReasonerModel.training_step` usage ✓
- `EvidenceAugmentedLLM.forward(E_c, edge_valid, v_src_c, v_tgt_c, qa_pairs)` — matches `LLMReasonerModel.training_step` call ✓
- `MemoryAugmentedLayer._G_LLM` attribute — set/cleared consistently in `EvidenceAugmentedLLM._set_G` / `_clear_G` ✓

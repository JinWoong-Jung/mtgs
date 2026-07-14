"""WP1 — GraphFeatureBundle: the frozen MTGS+GazeGraphBlock evidence for one clip.

All tensors follow the conventions in conventions.py:
  * matrices [.., looker_idx, target_idx]  (lah/laeo/sa/alignment/overlap)
  * person axis N is right-aligned padded (valid_person_mask marks the real ones)
  * edge target axis is N+2: [0..N-1] persons, [N] null_in, [N+1] null_out
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class GraphFeatureBundle:
    sample_ids: list          # [B] stable ids (dataset, center-frame path) hashes

    # ── node / edge states ──────────────────────────────────────────────────────
    person_tokens: torch.Tensor       # [B,T,N,512]  scene-contextualised person tokens
    v_src: torch.Tensor               # [B,T,N,De]   source node proj
    v_tgt: torch.Tensor               # [B,T,N+2,De] target node proj (persons + null_in/out)
    edge_states: torch.Tensor         # [B,T,N,N+2,De] refined edge tensor E[looker,target]

    # ── social readout logits (center frame in matrices; *_frames optional) ─────
    lah_logits: torch.Tensor          # [B,T,N,N]  logit(looker looks at target)
    laeo_logits: torch.Tensor         # [B,T,N,N]  symmetric
    sa_logits: torch.Tensor           # [B,T,N,N]  symmetric
    null_in_logits: torch.Tensor      # [B,T,N]
    null_out_logits: torch.Tensor     # [B,T,N]

    # ── per-person gaze evidence ────────────────────────────────────────────────
    gaze_vectors: torch.Tensor        # [B,T,N,2]
    gaze_heatmaps: torch.Tensor       # [B,T,N,H,W]
    inout_logits: torch.Tensor        # [B,T,N]
    head_bboxes: torch.Tensor         # [B,T,N,4]

    # ── geometric edge priors ───────────────────────────────────────────────────
    alignment: torch.Tensor           # [B,T,N,N]  cos(gaze[looker], center[looker]-center[target])
    overlap: torch.Tensor             # [B,T,N,N]  heatmap-in-box overlap

    # ── masks ───────────────────────────────────────────────────────────────────
    valid_person_mask: torch.Tensor   # [B,T,N]   bool
    pair_mask: torch.Tensor           # [B,T,N,N] bool (valid looker & target & not self)

    T_center: int = 2                 # center-frame index (temporal_context=2 -> T=5, center=2)

    # ── helpers ─────────────────────────────────────────────────────────────────

    def to(self, device):
        for f in fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v):
                setattr(self, f.name, v.to(device))
        return self

    def center(self):
        """Dict of the center-frame slices used to build pair queries."""
        c = self.T_center
        return {
            "person_tokens": self.person_tokens[:, c],   # [B,N,512]
            "v_src": self.v_src[:, c],                    # [B,N,De]
            "v_tgt": self.v_tgt[:, c],                    # [B,N+2,De]
            "edge_states": self.edge_states[:, c],        # [B,N,N+2,De]
            "lah_logits": self.lah_logits[:, c],          # [B,N,N]
            "laeo_logits": self.laeo_logits[:, c],
            "sa_logits": self.sa_logits[:, c],
            "null_in_logits": self.null_in_logits[:, c],  # [B,N]
            "null_out_logits": self.null_out_logits[:, c],
            "inout_logits": self.inout_logits[:, c],
            "alignment": self.alignment[:, c],            # [B,N,N]
            "overlap": self.overlap[:, c],
            "valid_person_mask": self.valid_person_mask[:, c],   # [B,N]
            "pair_mask": self.pair_mask[:, c],            # [B,N,N]
        }

    @property
    def shapes(self):
        return {f.name: tuple(getattr(self, f.name).shape)
                for f in fields(self) if torch.is_tensor(getattr(self, f.name))}


def bundle_from_cache_entry(sid: str, e: dict) -> "GraphFeatureBundle":
    """Build a B=1, T=1 (center-frame) GraphFeatureBundle from an existing vlmgraph_*.pt
    cache entry — NO re-extraction. edge_states is reassembled from edge_pp/null_in/out.
    Used for the WP3 single-frame external-residual gate (full T=5 needs a re-export)."""
    f = lambda x: x.float()
    N = e["head_bboxes"].shape[0]
    De = e["v_src"].shape[-1]
    # edge_states [N, N+2, De] = [edge_pp | edge_null_in | edge_null_out]
    edge = torch.cat([f(e["edge_pp"]),
                      f(e["edge_null_in"]).unsqueeze(1),
                      f(e["edge_null_out"]).unsqueeze(1)], dim=1)          # [N, N+2, De]
    vis = e["vis_mask"].bool() if "vis_mask" in e else e["person_mask"].bool()
    eye = torch.eye(N, dtype=torch.bool)
    pair = vis.unsqueeze(1) & vis.unsqueeze(0) & ~eye                       # [N,N] looker&target&¬self
    B1T1 = lambda t: t.unsqueeze(0).unsqueeze(0)                           # [.] -> [1,1,.]
    # v_tgt in cache is [N+2, De] already (persons + null_in/out)
    return GraphFeatureBundle(
        sample_ids=[sid],
        person_tokens=B1T1(torch.zeros(N, 512)),   # not cached per-entry; unused by decoder queries
        v_src=B1T1(f(e["v_src"])),
        v_tgt=B1T1(f(e["v_tgt"])),
        edge_states=B1T1(edge),
        lah_logits=B1T1(f(e["lah_logits"])),
        laeo_logits=B1T1(f(e["laeo_logits"])),
        sa_logits=B1T1(f(e["sa_logits"])),
        null_in_logits=B1T1(f(e["null_in_logits"])),
        null_out_logits=B1T1(f(e["null_out_logits"])),
        gaze_vectors=B1T1(f(e["gaze_vecs"])),
        gaze_heatmaps=B1T1(f(e["gaze_heatmap"])),
        inout_logits=B1T1(torch.zeros(N)),         # predicted inout not cached; unused by queries
        head_bboxes=B1T1(f(e["head_bboxes"])),
        alignment=B1T1(f(e["align"])),
        overlap=B1T1(f(e["overlap"])),
        valid_person_mask=B1T1(vis),
        pair_mask=B1T1(pair),
        T_center=0,
    )

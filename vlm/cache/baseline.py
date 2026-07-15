"""Export "original MTGS" (gaze_graph.use=False: decoder_lah/decoder_sa, LAEO=min(LAH,LAH^T))
per-pair logits, matching the same sid ordering as vlm/graph_export.py's vlmgraph_*.pt cache
so both can be sampled by the identical VLM manifest via vlm.social.input.sample_graph_logit.

Unlike vlm/graph_export.py (which requires cfg.gaze_graph.use=True and reads the
GazeGraphBlock's stashed _feat), this checkpoint has no gaze_graph_block submodule at all:
model(batch) returns (None, gaze_vec, gaze_hm, inout, lah, laeo, coatt, None, None) directly,
where lah/laeo/coatt are flat per-pair vectors in itertools.permutations(range(n), 2) order
(mtgs_net.py's `_pair_indices_cache`), matching the *_labels GT convention exactly.
"""
import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from tqdm import tqdm
from mtgs.networks.models import MTGSModel
from mtgs.train.dataset import build_dataset
from vlm.cache.config import make_cfg
from vlm.cache.matrix import pair_vector_to_matrix

STAGE   = {"train": "fit",   "val": "validate",     "test": "test"}
ATTR    = {"train": "train_dataset", "val": "val_dataset", "test": "test_dataset"}
DLOADER = {"train": "train_dataloader", "val": "val_dataloader", "test": "test_dataloader"}


def _to_dev(o, d):
    if torch.is_tensor(o): return o.to(d)
    if isinstance(o, dict): return {k: _to_dev(v, d) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return type(o)(_to_dev(v, d) for v in o)
    return o


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=4000)
    ap.add_argument("--num_people", default="all", help="'all' matches vlm/graph_export.py's offline-extraction ordering")
    ap.add_argument(
        "--indices_file", default="",
        help="optional text file of global dataset indices (one per line) to restrict "
             "export to -- sid is set to the GLOBAL index (matches vlm/data_prep.py's "
             "sid=f'sample{idx:06d}' convention with idx=indices[local_idx]), not a "
             "compact running counter, so the cache stays correctly aligned even with gaps.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_cfg(args.split, num_people=(args.num_people or None), use_graph=False)
    cfg.test.batch_size = args.batch_size
    if (cfg.data.num_people == "all" or args.split == "test") and args.batch_size != 1:
        print(f"[export-baseline] num_people=all -> forcing batch_size 1 (was {args.batch_size})", flush=True)
        args.batch_size = 1
        cfg.test.batch_size = 1

    # Same seed/ordering discipline as vlm/graph_export.py, required for sid alignment
    # with data_prep.py's overlay/manifest pass and the existing vlmgraph_*.pt cache.
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    data = build_dataset(**cfg); data.setup(STAGE[args.split])
    if args.split == "train":
        eval_tf = data.val_dataset.datasets[0].transform
        for sub in data.train_dataset.datasets:
            sub.split = "val"; sub.transform = eval_tf
    ds = getattr(data, ATTR[args.split])
    collate = getattr(data, DLOADER[args.split])().collate_fn

    global_indices = None
    if args.indices_file:
        with open(args.indices_file) as fh:
            global_indices = [int(line) for line in fh if line.strip()]
        if len(global_indices) != len(set(global_indices)):
            raise ValueError("--indices_file contains duplicate indices")
        if sorted(global_indices) != global_indices:
            raise ValueError("--indices_file must be sorted ascending (sid=global index contract)")
        if args.batch_size != 1:
            print("[export-baseline] --indices_file forces batch_size=1 (B>1 sid mapping "
                  "would be ambiguous)", flush=True)
        ds = torch.utils.data.Subset(ds, global_indices)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
        print(f"[export-baseline] restricted to {len(global_indices)} explicit global indices "
              f"(min={global_indices[0]} max={global_indices[-1]})", flush=True)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model = MTGSModel(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck["state_dict"], strict=False)
    if miss or unexp:
        raise RuntimeError(
            f"original-MTGS checkpoint did not load cleanly: "
            f"missing={miss[:10]} unexpected={unexp[:10]}"
        )
    print(f"[export-baseline] ckpt epoch={ck.get('epoch')} loaded cleanly "
          f"(0 missing, 0 unexpected, gaze_graph.use=False)", flush=True)
    model.eval().to(device)

    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" \
        else torch.autocast("cpu", enabled=False)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}; sample_idx = 0

    for local_idx, batch in enumerate(tqdm(loader, desc=f"export-baseline:{args.split}")):
        batch = _to_dev(batch, device)
        with autocast:
            outputs = model(batch)
        _, _gaze_vec, _gaze_hm, _inout, lah, laeo, coatt, null_in, null_out = outputs
        assert null_in is None and null_out is None, \
            "gaze_graph.use=False must not produce null edges"

        n = batch["head_bboxes"].shape[2]
        c = lah.shape[1] // 2  # center frame, matching vlm/graph_export.py

        lah_c   = lah[:, c].float().cpu()      # (B, P) P = n*(n-1)
        laeo_c  = laeo[:, c].float().cpu()
        coatt_c = coatt[:, c].float().cpu()

        bbc = batch["head_bboxes"][:, c].float().cpu()          # (B, n, 4)
        nvp = batch["num_valid_people"][:, c].long().cpu()      # (B,)
        ar = torch.arange(n).view(1, n)
        pmask = ar >= (n - nvp.view(-1, 1))                     # (B, n) trailing-valid convention
        area = (bbc[..., 2]-bbc[..., 0]).clamp_min(0) * (bbc[..., 3]-bbc[..., 1]).clamp_min(0)
        vis = pmask & (area > 1e-4)

        # LAH: model output matches the *_labels TARGET,LOOKER pair-vector convention
        # exactly (mtgs_net.py: "matching label order") -> reverse=True gives
        # lah_mat[a,b] = P(a looks at b), the same convention used by vlmgraph_*.pt.
        lah_mat  = pair_vector_to_matrix(lah_c, n, reverse=True)
        # LAEO/SA: inherently symmetric (LAEO by construction=min(lah,lah^T); SA's raw
        # per-direction decoder output is filled at both (i,j) and (j,i) here, exactly
        # like the GT conversion in vlm/graph_export.py) -> reverse doesn't matter.
        laeo_mat = pair_vector_to_matrix(laeo_c, n, reverse=False)
        sa_mat   = pair_vector_to_matrix(coatt_c, n, reverse=False)

        # GT, converted the same way, kept only for a cheap sid/label sanity cross-check
        # against the existing vlmgraph_*.pt cache (same frames, same manifest labels).
        lah_gt  = pair_vector_to_matrix(batch["lah_labels"][:, c, :].float().cpu(), n, reverse=True)
        laeo_gt = pair_vector_to_matrix(batch["laeo_labels"][:, c, :].float().cpu(), n, reverse=False)
        sa_gt   = pair_vector_to_matrix(batch["coatt_labels"][:, c, :].float().cpu(), n, reverse=False)

        B = lah_c.shape[0]
        for b in range(B):
            if global_indices is not None:
                assert B == 1, "batch_size must be 1 when --indices_file is set"
                sid = f"sample{global_indices[local_idx]:06d}"
            else:
                sid = f"sample{sample_idx + b:06d}"
            cache[sid] = {
                "lah_logits": lah_mat[b].half(),
                "laeo_logits": laeo_mat[b].half(),
                "sa_logits": sa_mat[b].half(),
                "head_bboxes": bbc[b],
                "vis_mask": vis[b],
                "person_mask": pmask[b],
                "num_persons": int(pmask[b].sum()),
                "lah_gt": lah_gt[b].long(),
                "laeo_gt": laeo_gt[b].long(),
                "sa_gt": sa_gt[b].long(),
            }
        sample_idx += B
        if args.limit and sample_idx >= args.limit: break
        if len(cache) and len(cache) % args.save_every < B: torch.save(cache, out_path)

    torch.save(cache, out_path)
    print(f"[export-baseline] saved {len(cache)} samples -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

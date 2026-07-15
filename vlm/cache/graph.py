import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")   # "resize storage not resizable" 회피
from torch.utils.data import DataLoader
from tqdm import tqdm
from mtgs.networks.models import MTGSModel
from mtgs.train.dataset import build_dataset
from vlm.cache.config import make_cfg
from vlm.cache.matrix import pair_vector_to_matrix
from vlm.cache.selection import apply_plan, load_plan

# 내 build_dataset 은 test_split 을 무시한다 → split 은 setup(stage)+attr 로 선택.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=4000)
    ap.add_argument("--num_people", default="", help="all or an integer N; empty keeps config default")
    ap.add_argument(
        "--selection_plan", default="",
        help="frozen vlm.cache.selection JSON; required for deterministic N-capped train/val export",
    )
    ap.add_argument(
        "--laeo_derive", choices=("decoder", "lah_min"), default="decoder",
        help="LAEO readout used for exported logits; decoder is required for V18 extraction",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested_people = None if not args.num_people else ("all" if args.num_people == "all" else int(args.num_people))
    if args.selection_plan and (requested_people is None or requested_people == "all"):
        raise ValueError("--selection_plan requires an integer --num_people cap")
    cfg = make_cfg(args.split, num_people=requested_people)
    cfg.gaze_graph.laeo_derive = args.laeo_derive
    cfg.test.batch_size = args.batch_size
    # Variable-N (num_people="all") frames can't be stacked -> force batch_size 1.
    if (cfg.data.num_people == "all" or args.split == "test") and args.batch_size != 1:
        print(f"[export] num_people=all -> forcing batch_size 1 (was {args.batch_size})", flush=True)
        args.batch_size = 1
        cfg.test.batch_size = 1

    # sid = enumeration index. data_prep(Task 5)와 sid 정렬을 위해 build_dataset 전에
    # 동일 시드 고정 + train split 은 stochastic 증강 비활성화(eval transform)로 맞춘다.
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    data = build_dataset(**cfg); data.setup(STAGE[args.split])
    ds = getattr(data, ATTR[args.split])
    if args.selection_plan:
        apply_plan(ds, load_plan(args.selection_plan, split=args.split, num_people=int(cfg.data.num_people)))
    if args.split == "train":
        eval_tf = data.val_dataset.datasets[0].transform
        for sub in data.train_dataset.datasets:
            sub.split = "val"; sub.transform = eval_tf
    collate = getattr(data, DLOADER[args.split])().collate_fn
    # num_workers=0: guarantees deterministic sample content per index, so sids
    # align exactly with data_prep.py's overlay/manifest pass. VSGaze __getitem__
    # uses numpy/python `random` (people-subset) that PyTorch doesn't per-worker
    # seed; nw>0 can desync sid->sample. Runtime here is graph-forward-bound, so
    # single-worker loading costs little. (--num_workers kept for API compat.)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

    model = MTGSModel(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck["state_dict"], strict=False)
    laeo_load_issues = [key for key in [*miss, *unexp] if "head_laeo" in key]
    if args.laeo_derive == "decoder" and laeo_load_issues:
        raise RuntimeError(
            f"V18 LAEO decoder weights did not load cleanly: {laeo_load_issues[:10]}"
        )
    print(f"[export] ckpt epoch={ck.get('epoch')} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval().to(device)
    model.model.gaze_graph_block.export_features = True

    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" \
        else torch.autocast("cpu", enabled=False)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}; sample_idx = 0

    for batch in tqdm(loader, desc=f"export:{args.split}"):
        batch = _to_dev(batch, device)
        with autocast:
            model(batch)
        f = model.model.gaze_graph_block._feat
        B, T, N = f["lah_mat"].shape[:3]; c = T // 2; De = f["v_src"].shape[-1]
        E = f["E"]                                             # (B,T,N,N+2,De)
        lah_c = f["lah_mat"][:, c].float().cpu()               # (B,N,N)
        sa_c  = f["sa_mat"][:, c].float().cpu()
        # LAEO: lah_min 모드면 laeo_mat=None → min(lah, lah^T) (logit space, 대칭)
        if f["laeo_mat"] is None:
            if args.laeo_derive == "decoder":
                raise RuntimeError(
                    "LAEO decoder export requested but gaze_graph_block produced no laeo_mat"
                )
            laeo_all = torch.minimum(f["lah_mat"], f["lah_mat"].transpose(-1, -2))
        else:
            laeo_all = f["laeo_mat"]
        laeo_c = laeo_all[:, c].float().cpu()

        # GT ([i,j]="i looks at j"): lah reverse=True (peer 규약)
        lah_gt  = pair_vector_to_matrix(batch["lah_labels"][:, c, :],  N, reverse=True).cpu()
        laeo_gt = pair_vector_to_matrix(batch["laeo_labels"][:, c, :], N).cpu()
        sa_gt   = pair_vector_to_matrix(batch["coatt_labels"][:, c, :], N).cpu()
        inout_gt = batch["inout"][:, c, :].cpu()
        bbc = batch["head_bboxes"][:, c].float().cpu()         # (B,N,4)
        nvp = batch["num_valid_people"][:, c].long().cpu()     # (B,)
        ar = torch.arange(N).view(1, N)
        pmask = ar >= (N - nvp.view(-1, 1))                    # (B,N) 뒤쪽 valid
        area = (bbc[..., 2]-bbc[..., 0]).clamp_min(0) * (bbc[..., 3]-bbc[..., 1]).clamp_min(0)
        vis = pmask & (area > 1e-4)

        for b in range(B):
            sid = f"sample{sample_idx + b:06d}"
            cache[sid] = {
                "lah_logits": lah_c[b].half(), "laeo_logits": laeo_c[b].half(), "sa_logits": sa_c[b].half(),
                "null_in_logits": f["null_in"][b, c].float().cpu().half(),
                "null_out_logits": f["null_out"][b, c].float().cpu().half(),
                "lah_logits_frames":  f["lah_mat"][b].float().cpu(),
                "laeo_logits_frames": laeo_all[b].float().cpu(),
                "sa_logits_frames":   f["sa_mat"][b].float().cpu(),
                "null_in_frames":  f["null_in"][b].float().cpu(),
                "null_out_frames": f["null_out"][b].float().cpu(),
                "v_src": f["v_src"][b, c].half().cpu(), "v_tgt": f["v_tgt"][b, c].half().cpu(),
                "edge_pp":       E[b, c, :, :N, :].half().cpu(),
                "edge_null_in":  E[b, c, :, N,  :].half().cpu(),
                "edge_null_out": E[b, c, :, N+1, :].half().cpu(),
                "align":   f["align"][b, c].half().cpu(),
                "overlap": f["overlap"][b, c].half().cpu(),
                "gaze_vecs":  f["gaze_vecs"][b, c].float().cpu(),
                "gaze_point": f["gaze_point"][b, c].float().cpu(),
                "gaze_heatmap": f["gaze_heatmap"][b, c].half().cpu(),   # (N,Hh,Ww) for <hmtok>
                "head_bboxes": bbc[b],
                "lah_gt": lah_gt[b], "laeo_gt": laeo_gt[b], "sa_gt": sa_gt[b], "inout_gt": inout_gt[b],
                "person_mask": pmask[b], "vis_mask": vis[b], "num_persons": int(pmask[b].sum()),
            }
        sample_idx += B
        if args.limit and sample_idx >= args.limit: break
        if len(cache) and len(cache) % args.save_every < B: torch.save(cache, out_path)

    torch.save(cache, out_path)
    print(f"[export] saved {len(cache)} samples -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

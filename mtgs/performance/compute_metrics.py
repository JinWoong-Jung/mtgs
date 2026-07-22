# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import io
import itertools
import pickle

import torch
import random

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

from mtgs.performance.metrics import GFTestDistance

import logging

logger = logging.getLogger(__name__)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        else:
            return super().find_class(module, name)


# ── Post-Processing helpers ────────────────────────────────────────────────

def _is_inside(head_bbox, gaze_pt):
    return (gaze_pt[0] > head_bbox[0] and gaze_pt[0] < head_bbox[2] and
            gaze_pt[1] > head_bbox[1] and gaze_pt[1] < head_bbox[3])


def _process_lah_pp(res):
    """Geometric LAH PP: gaze_point of person i falls inside head_bbox of person j."""
    head_boxes = res["head_bboxes"].numpy()
    N = head_boxes.shape[0]
    gaze_preds = res["gp_pred"].numpy()
    inout = res["inout_gt"].numpy()
    lah_gt = res["lah_gt"].numpy()
    pairs = list(itertools.permutations(range(N), 2))
    pair_arr = [list(p) for p in pairs]

    gts, preds = [], []
    for p in range(N):
        is_gf = res["dataset"][0] == "gazefollow"
        pio = 1 if is_gf else int(inout[p])
        gpred = gaze_preds if is_gf else gaze_preds[p]
        if is_gf and p != N - 1:
            preds.append(0); gts.append(-1); continue
        if pio == 1:
            pred, gt = 0, -1
            valid = [i for i, (src, tgt) in enumerate(pair_arr)
                     if tgt == p and src != 0]
            valid_lah = [lah_gt[i] for i in valid]
            if any(v != -1 for v in valid_lah):
                pos = [i for i, v in zip(valid, valid_lah) if v == 1]
                if pos:
                    gt = 1
                    src_person = pair_arr[pos[0]][0]
                    pred = 1 if _is_inside(head_boxes[src_person], gpred) else 0
                else:
                    gt = 0
                    pred = int(any(_is_inside(head_boxes[h], gpred)
                                   for h in range(N) if h != p))
            preds.append(pred); gts.append(gt)
    return gts, preds


def _process_laeo_pp(res):
    """Geometric LAEO PP: mutual gaze_point inside head_bbox."""
    head_boxes = res["head_bboxes"].numpy()
    N = head_boxes.shape[0]
    gaze_preds = res["gp_pred"].numpy()
    laeo_gt = res["laeo_gt"].tolist()
    pairs = list(itertools.permutations(range(N), 2))

    gts, preds = [], []
    for pid, (i, j) in enumerate(pairs):
        if laeo_gt[pid] == -1:
            continue
        pred = 1 if (_is_inside(head_boxes[i], gaze_preds[j]) and
                     _is_inside(head_boxes[j], gaze_preds[i])) else 0
        preds.append(pred); gts.append(laeo_gt[pid])
    return gts, preds


# ── Main compute function ──────────────────────────────────────────────────

def compute(results, dataset=None, shuffle=False, thr=0.5):
    """Compute all VSGaze metrics and return a dict of scalar values.

    Set ``dataset`` to one VSGaze source (childplay, videoattentiontarget,
    laeo, or coatt) to recompute the same metrics for that subset only.
    Tasks without valid labels are reported as N/A instead of raising.
    """
    gf_metrics = GFTestDistance()
    logger.info("Computing metrics...")
    if shuffle:
        random.shuffle(results)

    lah_gt_all, lah_pred_all = [], []
    laeo_gt_all, laeo_pred_all = [], []
    coatt_gt_all, coatt_pred_all = [], []
    distances, avg_distances = [], []
    inout_gt_all, inout_pred_all = [], []
    mask_all = []

    # PP accumulators
    lah_pp_gt_all, lah_pp_pred_all = [], []
    laeo_pp_gt_all, laeo_pp_pred_all = [], []

    for batch in tqdm(results):
        if dataset is not None and batch["dataset"][0] != dataset:
            continue

        # ── Distance (N/A for VLM-only test_predictions.p) ───────────────
        if "gp_pred" in batch:
            if batch["dataset"][0] == "gazefollow":
                test_dist_to_avg, _, test_min_dist = gf_metrics(
                    batch["gp_pred"].cpu(), batch["gp_gt"].cpu()
                )
                avg_distances.append(test_dist_to_avg.unsqueeze(0))
                distances.append(test_min_dist.unsqueeze(0))
            else:
                dist = (batch["gp_pred"] - batch["gp_gt"]).norm(2, dim=-1)
                distances.append(dist[batch["inout_gt"] == 1].cpu())

        # ── In-out (N/A for VLM-only test_predictions.p) ─────────────────
        if "inout_pred" in batch and batch["dataset"][0] in ["videoattentiontarget", "childplay"]:
            mask = batch["inout_gt"] != -1
            inout_gt_all.append(batch["inout_gt"][mask].cpu())
            inout_pred_all.append(batch["inout_pred"][mask].cpu())

        batch_size, num_people = batch["head_bboxes"].shape[:2]
        pair_indices = torch.tensor(
            list(itertools.permutations(torch.arange(num_people), 2))
        )

        # ── LAH (model score) ─────────────────────────────────────────────
        lah_gt = batch["lah_gt"].cpu()
        lah_pred = batch["lah_pred"].cpu()
        lah_gt_metric = torch.zeros(batch_size, num_people).long() - 1
        lah_pred_metric = torch.zeros(batch_size, num_people)
        for bi in range(batch_size):
            for pi in range(num_people):
                if batch["dataset"][0] == "gazefollow":
                    io = 1
                else:
                    io = batch["inout_gt"][bi][pi] == 1
                if io == 1:
                    valid_indices = torch.where((pair_indices[:, 1] == pi).int())[0]
                    if valid_indices.shape[0] > 0:
                        if (lah_gt[bi][valid_indices] != -1).sum() == 0:
                            continue
                        max_val, _ = torch.max(lah_pred[bi][valid_indices], 0)
                        lah_gt_metric[bi][pi] = min(
                            lah_gt[bi][valid_indices][
                                lah_gt[bi][valid_indices] != -1
                            ].sum(),
                            1,
                        )
                        gt_idx = torch.where(lah_gt[bi][valid_indices] == 1)[0]
                        if len(gt_idx) > 0:
                            gt_idx = gt_idx[0]
                            # Score the true-positive edge with the model's own
                            # confidence on it, even when some other edge outranks it.
                            lah_pred_metric[bi][pi] = lah_pred[bi][
                                valid_indices
                            ][gt_idx]
                        else:
                            lah_pred_metric[bi][pi] = max_val
        mask = lah_gt_metric != -1
        mask_all.append(mask[0])
        lah_gt_all.append(lah_gt_metric[0].cpu())
        lah_pred_all.append(lah_pred_metric[0].cpu())

        # ── LAEO (model score) ────────────────────────────────────────────
        laeo_gt = batch["laeo_gt"].cpu()
        mask = laeo_gt != -1
        if mask.sum() > 0:
            laeo_pred = batch["laeo_pred"]
            laeo_pred_argmax = torch.zeros_like(laeo_pred)
            for bi in range(batch_size):
                for pi in range(num_people):
                    valid_indices = torch.where(
                        (pair_indices[:, 1] == pi).int()
                        * (pair_indices[:, 0] != 0).int()
                    )[0]
                    if valid_indices.shape[0] > 0:
                        max_val, max_idx = torch.max(laeo_pred[bi][valid_indices], 0)
                        laeo_pred_argmax[bi][valid_indices[max_idx]] = max_val
            laeo_gt = laeo_gt[mask]
            laeo_pred_argmax = laeo_pred_argmax[mask]
            if len(laeo_gt) > 0:
                laeo_gt_all.append(laeo_gt.cpu())
                laeo_pred_all.append(laeo_pred_argmax.float().cpu())

        # ── CoAtt (model score) ───────────────────────────────────────────
        mask = batch["coatt_gt"] != -1
        batch_coatt_gt = batch["coatt_gt"][mask]
        batch_coatt_pred = batch["coatt_pred"][mask]
        if len(batch_coatt_gt) > 0:
            coatt_gt_all.append(batch_coatt_gt.cpu())
            coatt_pred_all.append(batch_coatt_pred.float().cpu())

        # ── PP geometric (N/A for VLM-only test_predictions.p) ───────────
        if "gp_pred" in batch:
            sq = {k: v.squeeze(0) if hasattr(v, "squeeze") else v
                  for k, v in batch.items()}
            g, p = _process_lah_pp(sq)
            lah_pp_gt_all.extend(g); lah_pp_pred_all.extend(p)

            g, p = _process_laeo_pp(sq)
            laeo_pp_gt_all.extend(g); laeo_pp_pred_all.extend(p)

    import numpy as np
    lah_pp_gt = np.array(lah_pp_gt_all)
    lah_pp_pred = np.array(lah_pp_pred_all)
    pp_mask = lah_pp_gt != -1
    laeo_pp_gt = np.array(laeo_pp_gt_all)
    laeo_pp_pred = np.array(laeo_pp_pred_all)

    ret = {}

    def _has_two_classes(labels):
        return labels.numel() > 0 and torch.unique(labels).numel() == 2

    def _log_binary_task(name, labels, scores, key_prefix):
        """Log threshold metrics for any non-empty task; AP/AUC need two classes."""
        logger.info("----- %s -----", name)
        labels = labels.reshape(-1)
        scores = scores.reshape(-1)
        if labels.numel() == 0:
            logger.info("N/A   : no valid labels")
            return

        if _has_two_classes(labels):
            ap = average_precision_score(labels.numpy(), scores.numpy())
            auc = roc_auc_score(labels.numpy(), scores.numpy())
            logger.info("AP    : %.4f", ap)
            logger.info("AUC   : %.4f", auc)
            ret[f"{key_prefix}_ap"] = ap
            ret[f"{key_prefix}_auc"] = auc
        else:
            logger.info("AP/AUC: N/A (single-class target)")

        pred_thr = scores > thr
        logger.info("Prec  : %.4f", precision_score(labels.numpy(), pred_thr.numpy(), zero_division=0))
        logger.info("Recall: %.4f", recall_score(labels.numpy(), pred_thr.numpy(), zero_division=0))
        logger.info(
            "F1    : %.4f  (thr=%.1f)",
            f1_score(labels.numpy(), pred_thr.numpy(), zero_division=0),
            thr,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Block 1: README-style metrics for the selected dataset scope
    # ══════════════════════════════════════════════════════════════════════
    scope = f"{dataset} subset" if dataset is not None else "VSGaze test set"
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PRIMARY METRICS  (%s)", scope)
    logger.info("=" * 60)

    if distances:
        dist_val = torch.cat(distances).mean().item()
        logger.info("Dist        : %.4f", dist_val)
        ret["dist"] = dist_val
    if avg_distances:
        logger.info("Avg Dist    : %.4f", torch.cat(avg_distances).mean().item())

    if inout_gt_all:
        inout_gt_cat = torch.cat(inout_gt_all).float()
        inout_pred_cat = torch.cat(inout_pred_all).float()
        if _has_two_classes(inout_gt_cat):
            ap_io = average_precision_score(inout_gt_cat.numpy(), inout_pred_cat.numpy())
            logger.info("AP_IO       : %.4f", ap_io)
            ret["ap_io"] = ap_io
        else:
            logger.info("AP_IO       : N/A (single-class target)")

    if lah_pp_gt_all and pp_mask.sum() > 0:
        f1_lah_pp = f1_score(lah_pp_gt[pp_mask], lah_pp_pred[pp_mask], zero_division=0)
        logger.info("F1_LAH (PP) : %.4f", f1_lah_pp)
        ret["f1_lah_pp"] = f1_lah_pp

    f1_laeo_pp = None
    if laeo_pp_gt_all and len(laeo_pp_gt) > 0 and laeo_pp_gt.sum() > 0 and laeo_pp_pred.sum() > 0:
        f1_laeo_pp = f1_score(laeo_pp_gt, laeo_pp_pred, zero_division=0)
        logger.info("F1_LAEO(PP) : %.4f", f1_laeo_pp)
        ret["f1_laeo_pp"] = f1_laeo_pp
    elif laeo_pp_gt_all:
        logger.info("F1_LAEO(PP) : N/A (no positive predictions)")

    coatt_gt_cat = coatt_pred_cat = None
    if coatt_gt_all:
        coatt_gt_cat = torch.cat(coatt_gt_all)
        coatt_pred_cat = torch.cat(coatt_pred_all)
        if _has_two_classes(coatt_gt_cat):
            ap_sa = average_precision_score(coatt_gt_cat.numpy(), coatt_pred_cat.numpy())
            logger.info("AP_SA       : %.4f", ap_sa)
            ret["coatt_ap"] = ap_sa
        else:
            logger.info("AP_SA       : N/A (single-class target)")

    # ══════════════════════════════════════════════════════════════════════
    # Block 2: task-specific metrics; unavailable task labels are N/A.
    # ══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("  DETAILED METRICS")
    logger.info("=" * 60)

    laeo_gt_cat = torch.cat(laeo_gt_all) if laeo_gt_all else torch.empty(0, dtype=torch.long)
    laeo_pred_cat = torch.cat(laeo_pred_all) if laeo_pred_all else torch.empty(0)
    _log_binary_task("LAEO", laeo_gt_cat, laeo_pred_cat, "laeo")
    if f1_laeo_pp is not None:
        logger.info("F1 PP : %.4f  (geometric)", f1_laeo_pp)
        logger.info("  Prec PP : %.4f", precision_score(laeo_pp_gt, laeo_pp_pred, zero_division=0))
        logger.info("  Rec  PP : %.4f", recall_score(laeo_pp_gt, laeo_pp_pred, zero_division=0))

    if lah_gt_all:
        lah_gt_cat = torch.cat(lah_gt_all)
        lah_pred_cat = torch.cat(lah_pred_all)
        mask_cat = torch.cat(mask_all)
        lah_gt_cat = lah_gt_cat[mask_cat]
        lah_pred_cat = lah_pred_cat[mask_cat]
    else:
        lah_gt_cat = torch.empty(0, dtype=torch.long)
        lah_pred_cat = torch.empty(0)
    _log_binary_task("LAH", lah_gt_cat, lah_pred_cat, "lah")
    if lah_pp_gt_all and pp_mask.sum() > 0:
        logger.info("F1 PP : %.4f  (geometric)", f1_lah_pp)
        logger.info("  Prec PP : %.4f", precision_score(lah_pp_gt[pp_mask], lah_pp_pred[pp_mask], zero_division=0))
        logger.info("  Rec  PP : %.4f", recall_score(lah_pp_gt[pp_mask], lah_pp_pred[pp_mask], zero_division=0))

    if coatt_gt_cat is None:
        coatt_gt_cat = torch.empty(0, dtype=torch.long)
        coatt_pred_cat = torch.empty(0)
    _log_binary_task("CoAtt (SA)", coatt_gt_cat, coatt_pred_cat, "coatt")

    ap_vals = [ret[k] for k in ("lah_ap", "laeo_ap", "coatt_ap") if k in ret]
    if ap_vals:
        ret["social_ap"] = sum(ap_vals) / len(ap_vals)

    return ret


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
    )

    pickle_path = '.../test_predictions.p'
    with open(pickle_path, 'rb') as f:
        unpickler = CPU_Unpickler(f)
        results = []
        while True:
            try:
                results.append(unpickler.load())
            except EOFError:
                break
    compute(results, shuffle=False, thr=0.5)

# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

import torch
import torch.nn.functional as F


def focal_social_loss(social_pred, social_gt, mask, gamma=2.0, pos_weight=2.0, label_smoothing=0.05):
    """Focal BCE for sparse social labels (e.g. LAEO mutual gaze).

    Combines three stabilization mechanisms:
    - Focal weighting (1-p_t)^gamma  — down-weights easy negatives that dominate LAEO batches
    - pos_weight                      — upweights rare positives (LAEO ≪ LAH in frequency)
    - label_smoothing                 — softens targets for noisy derived labels (LAEO=min(LAH_ij,LAH_ji))
    """
    social_gt = social_gt * mask
    gt_f = social_gt.float()

    # label smoothing: 0 → eps/2, 1 → 1 - eps/2
    gt_smooth = gt_f * (1.0 - label_smoothing) + 0.5 * label_smoothing

    with torch.no_grad():
        p = torch.sigmoid(social_pred)
        p_t = p * gt_f + (1.0 - p) * (1.0 - gt_f)
        focal_w = (1.0 - p_t) ** gamma

    loss = F.binary_cross_entropy_with_logits(
        social_pred,
        gt_smooth,
        pos_weight=torch.tensor(pos_weight, device=social_gt.device),
        reduction="none",
    )
    loss = focal_w * loss
    num_instances = mask.sum()
    return torch.mul(loss, mask).sum() / (num_instances + 1e-6)


def social_loss(social_pred, social_gt, mask, pos_weight=2.0):
    """Compute a loss for coatt or laeo or lah. This implements a standard binary cross-entropy loss.

    Args:
        social_pred: tensor representing the predicted social gaze logits.
        social_gt: tensor representing the ground-truth social gaze binary labels.
        mask: a binary tensor denoting the positions of valid predictions to keep in the loss. This is
        used to discard social gaze pairs where one side is a padded person (ie. black head image).

    Returns:
        Tensor representing the loss value (including the corresponding computation graph)
        Dictionary representing the items to log (e.g. {"total_loss": total_loss})
    """

    # set social_gt positions where mask is 0 to 0 (to avoid NaNs in loss computation)
    social_gt = social_gt * mask

    num_instances = mask.sum()
    loss = F.binary_cross_entropy_with_logits(
        social_pred,
        social_gt,
        pos_weight=torch.tensor(pos_weight, device=social_gt.device),
        reduction="none",
    )
    loss = torch.mul(loss, mask).sum() / (num_instances + 1e-6)

    return loss


def social_loss_prob(social_pred, social_gt, mask):
    """BCE loss for probability inputs in [0,1] (e.g. SA P@P^T output)."""
    social_gt = social_gt * mask
    num_instances = mask.sum()
    with torch.amp.autocast("cuda", enabled=False):
        loss = F.binary_cross_entropy(
            social_pred.float().clamp(1e-6, 1 - 1e-6),
            social_gt.float(),
            reduction="none",
        )
    return torch.mul(loss, mask).sum() / (num_instances + 1e-6)


def compute_social_loss(
    lah_pred,
    lah_gt,
    lah_mask,
    laeo_pred,
    laeo_gt,
    laeo_mask,
    coatt_pred,
    coatt_gt,
    coatt_mask,
    coatt_is_prob=False,
):
    lah_loss = social_loss(lah_pred, lah_gt, lah_mask, pos_weight=3.0)
    laeo_loss = focal_social_loss(laeo_pred, laeo_gt, laeo_mask)
    coatt_loss = (
        social_loss_prob(coatt_pred, coatt_gt, coatt_mask)
        if coatt_is_prob
        else social_loss(coatt_pred, coatt_gt, coatt_mask)
    )

    lah_coeff = 1
    laeo_coeff = 1
    coatt_coeff = 1
    total_loss = (
        lah_coeff * lah_loss + laeo_coeff * laeo_loss + coatt_coeff * coatt_loss
    )

    logs = {
        "laeo_loss": laeo_loss.item(),
        "lah_loss": lah_loss.item(),
        "coatt_loss": coatt_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_interact_loss(
    gaze_vec_gt,
    gaze_hm_gt,
    inout_gt,
    gaze_vec_pred,
    gaze_hm_pred,
    inout_pred,
    dataset=None,
    epoch=None,
):
    heatmap_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    angular_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    dist_loss = torch.tensor(0.0).to(gaze_hm_pred.device)
    inout_loss = torch.tensor(0.0).to(gaze_hm_pred.device)

    mask = inout_gt == 1
    # to avoid case where all samples of the batch are outside (i.e. division by 0)
    if torch.sum(mask) > 0:
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, mask, dataset)
        heatmap_loss = compute_heatmap_loss(gaze_hm_pred, gaze_hm_gt, mask, dataset)
    #         dist_loss = compute_dist_loss(gaze_pt_pred, gaze_pt_gt, mask)

    mask = inout_gt != -1
    if torch.sum(mask) > 0:
        inout_loss = compute_inout_loss(inout_pred, inout_gt, mask)
    #     total_loss = 20 * angular_loss + 1000 * heatmap_loss    # for GeomGaze
    total_loss = (
        3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss + 2 * inout_loss
    )  # for GazeInteract

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "dist_loss": dist_loss.item(),
        "inout_loss": inout_loss.item(),
        "angular_loss": angular_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_sharingan_loss(
    gaze_vec_gt,
    gaze_pt_gt,
    inout_gt,
    gaze_vec_pred,
    gaze_pt_pred,
    inout_pred,
    epoch=None,
):
    heatmap_loss = torch.tensor(0.0)
    angular_loss = torch.tensor(0.0)
    dist_loss = torch.tensor(0.0)
    inout_loss = torch.tensor(0.0)

    mask = inout_gt == 1
    # to avoid case where all samples of the batch are outside (i.e. division by 0)
    if torch.sum(mask) > 0:
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, mask)
        dist_loss = compute_dist_loss(gaze_pt_pred, gaze_pt_gt, mask)

    mask = inout_gt != -1
    if torch.sum(mask) > 0:
        inout_loss = compute_inout_loss(inout_pred, inout_gt, mask)
    total_loss = (
        3 * angular_loss + 100 * dist_loss + 1000 * heatmap_loss + 2 * inout_loss
    )

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "dist_loss": dist_loss.item(),
        "inout_loss": inout_loss.item(),
        "angular_loss": angular_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_dist_loss(gp_pred, gp_gt, mask):
    dist_loss = (gp_pred - gp_gt).pow(2).sum(dim=-1)
    dist_loss = torch.mul(dist_loss, mask)
    dist_loss = torch.sum(dist_loss) / torch.sum(mask)
    return dist_loss


def compute_heatmap_loss(hm_pred, hm_gt, mask, dataset=None):
    heatmap_loss = F.mse_loss(hm_pred, hm_gt, reduce=False).mean([2, 3])
    heatmap_loss = torch.mul(heatmap_loss, mask)
    if dataset:
        dataset_mask = np.where(
            (np.array(dataset) == "coatt").astype(np.int)
            + (np.array(dataset) == "laeo").astype(np.int)
        )[0]
        fact = torch.zeros_like(heatmap_loss) + 1
        fact[dataset_mask] = 0.1  # 0.1x loss for UCO-LAEO and VideoCoAtt
        heatmap_loss = heatmap_loss * fact
    heatmap_loss = torch.sum(heatmap_loss) / torch.sum(mask)
    return heatmap_loss


def compute_angular_loss(gv_pred, gv_gt, mask, dataset=None):
    angular_loss = (1 - (gv_pred * gv_gt).sum(axis=-1)) / 2
    angular_loss = torch.mul(angular_loss, mask)
    if dataset:
        dataset_mask = np.where(
            (np.array(dataset) == "coatt").astype(np.int)
            + (np.array(dataset) == "laeo").astype(np.int)
        )[0]
        fact = torch.zeros_like(angular_loss) + 1
        fact[dataset_mask] = 0.1  # 0.1x loss for UCO-LAEO and VideoCoAtt
        angular_loss = angular_loss * fact
    angular_loss = torch.sum(angular_loss) / torch.sum(mask)
    return angular_loss


def compute_inout_loss(io_pred, io_gt, mask):
    io_gt = io_gt * mask
    bce_loss = F.binary_cross_entropy_with_logits(io_pred, io_gt, reduction="none")
    bce_loss = (bce_loss * mask).sum() / mask.sum()

    return bce_loss


def compute_dual_null_loss(alpha_null_out, alpha_null_in, inout_gt, lah_gt, num_valid_people):
    """Dual-null supervision for SocialGraphBlock.

    L_null_out: BCE(alpha_null_out[g], inout_gt[g] == 0)
        Out-of-frame person should route attention to Null_out.
        Computed for every valid person with a known inout label.

    L_null_in: BCE(alpha_null_in[g], all(lah_gt[g→j] == 0) AND inout_gt[g] == 1)
        In-frame person who looks at nobody should route attention to Null_in.
        Computed only when inout_gt==1 and at least one LAH annotation exists.

    Valid people occupy BACK slots [N-nv .. N-1] per the padding convention.
    LAH edges are ordered as itertools.permutations(range(N), 2), so pair
    (g, d) is at position g*(N-1) + (d if d < g else d-1).

    Args:
        alpha_null_out:   (BT, N) in [0,1] — probability of routing to Null_out
        alpha_null_in:    (BT, N) in [0,1] — probability of routing to Null_in
        inout_gt:         (BT, N) — 0=out-of-frame, 1=in-frame, -1=unknown
        lah_gt:           (BT, N*(N-1)) — GT LAH labels (0 / 1 / -1)
        num_valid_people: (BT,) int

    Returns:
        (loss_null_out, loss_null_in) — scalar BCE losses.
    """
    device = alpha_null_out.device
    BT = alpha_null_out.shape[0]
    N  = alpha_null_out.shape[1]

    out_probs, out_targets = [], []
    in_probs,  in_targets  = [], []

    for bt in range(BT):
        nv = int(num_valid_people[bt])
        if nv == 0:
            continue
        nv_start = N - nv  # valid people at global positions nv_start..N-1

        for i_local in range(nv):
            g  = nv_start + i_local
            io = inout_gt[bt, g].item()

            # ── L_null_out ────────────────────────────────────────────────────
            # Every valid person with a known inout label contributes.
            if io != -1:
                out_probs.append(alpha_null_out[bt, g])
                out_targets.append(1.0 if io == 0 else 0.0)

            # ── L_null_in ─────────────────────────────────────────────────────
            if io != 1:
                continue
            if nv <= 1:
                # Only valid person in frame → must be looking at scene object
                in_probs.append(alpha_null_in[bt, g])
                in_targets.append(1.0)
                continue
            # Dataset convention: pair (a,b) = label "b looks at a" (TARGET, LOOKER).
            # "g looks at d" is stored at pair (d, g) → flat index d*(N-1)+(g if g<d else g-1)
            outgoing = [
                d * (N - 1) + (g if g < d else g - 1)
                for d in range(nv_start, N) if d != g
            ]
            lah_i = lah_gt[bt, outgoing]
            known_lah = lah_i[lah_i != -1]
            if len(known_lah) == 0:
                continue  # no annotation for this person
            null_in_gt = 0.0 if (known_lah == 1).any() else 1.0
            in_probs.append(alpha_null_in[bt, g])
            in_targets.append(null_in_gt)

    with torch.amp.autocast('cuda', enabled=False):
        if out_probs:
            probs_t   = torch.stack(out_probs).float()
            targets_t = torch.tensor(out_targets, dtype=torch.float32, device=device)
            loss_out  = F.binary_cross_entropy(probs_t, targets_t)
        else:
            loss_out  = torch.tensor(0.0, device=device, requires_grad=True)

        if in_probs:
            probs_t   = torch.stack(in_probs).float()
            targets_t = torch.tensor(in_targets, dtype=torch.float32, device=device)
            loss_in   = F.binary_cross_entropy(probs_t, targets_t)
        else:
            loss_in   = torch.tensor(0.0, device=device, requires_grad=True)

    return loss_out, loss_in

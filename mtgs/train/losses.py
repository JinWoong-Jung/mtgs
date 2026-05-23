# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

import torch
import torch.nn.functional as F


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

    # Intersect annotation mask with finiteness: skip positions where the logit is
    # non-finite (safety guard — root cause is dataset padding pid -2, but keep as
    # defence against any future annotation/masking mismatches).
    finite_mask = mask & torch.isfinite(social_pred)
    num_instances = finite_mask.sum()
    if num_instances == 0:
        # No valid pairs: return a plain zero — avoids NaN from empty reduction.
        return torch.tensor(0.0, device=social_pred.device)

    # Index only valid, finite positions before BCE.
    valid_pred = social_pred[finite_mask]
    valid_gt = social_gt[finite_mask].float()
    loss = F.binary_cross_entropy_with_logits(
        valid_pred,
        valid_gt,
        pos_weight=torch.tensor(pos_weight, device=valid_gt.device),
        reduction="sum",
    )
    return loss / num_instances.float()


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
):
    lah_loss = social_loss(lah_pred, lah_gt, lah_mask, pos_weight=3.0)
    laeo_loss = social_loss(laeo_pred, laeo_gt, laeo_mask)
    coatt_loss = social_loss(coatt_pred, coatt_gt, coatt_mask)

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


def compute_null_node_loss(null_logits, lah_gt, num_valid_people):
    """Null-node supervision for SocialGraphBlock.

    For each valid source person g:
      null_GT(g) = 1  if ALL lah_gt(g→j) == 0  (person g looks at nobody)
      null_GT(g) = 0  if ANY lah_gt(g→j) == 1  (person g looks at someone)
      skip            if ALL lah_gt(g→j) == -1  (no annotation for this person)

    Valid people occupy the BACK slots (global positions N-nv..N-1) per the
    dataset's prepend-zero padding convention. Edges in lah_gt are ordered as
    itertools.permutations(range(N), 2), so pair (g, d) is at position
    g*(N-1) + (d if d < g else d-1).

    Args:
        null_logits:       (BT, N)         source-to-null logits
        lah_gt:            (BT, N*(N-1))   GT LAH labels (0 / 1 / -1)
        num_valid_people:  (BT,) int       valid person count per frame

    Returns:
        Scalar BCE loss over all annotated (sample, person) pairs.
    """
    device = null_logits.device
    BT = null_logits.shape[0]
    N = null_logits.shape[1]

    valid_logits = []
    valid_gts = []

    for bt in range(BT):
        nv = int(num_valid_people[bt])
        if nv <= 1:
            continue
        nv_start = N - nv  # valid people at global positions nv_start..N-1
        for i_local in range(nv):
            g = nv_start + i_local  # global position of this valid person
            # Outgoing edge positions from g to other valid people
            outgoing = [
                g * (N - 1) + (d if d < g else d - 1)
                for d in range(nv_start, N) if d != g
            ]
            lah_i = lah_gt[bt, outgoing]

            if (lah_i == -1).all():
                continue

            null_gt = 0.0 if (lah_i == 1).any() else 1.0
            valid_logits.append(null_logits[bt, g])
            valid_gts.append(null_gt)

    if not valid_logits:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_t = torch.stack(valid_logits)
    gts_t = torch.tensor(valid_gts, dtype=logits_t.dtype, device=device)
    return F.binary_cross_entropy_with_logits(logits_t, gts_t)

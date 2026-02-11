# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import List

import itertools
import matplotlib.cm as cm

import numpy as np

import torch
import torchvision.transforms.functional as TF

from mtgs.utils.image import (
    draw_arrowed_line,
    draw_rectangle,
    get_text_size,
    draw_circle,
    draw_text,
    draw_line,
)


def draw_gaze(
    image: np.ndarray,
    social_preds: List,
    head_bboxes: torch.Tensor,
    gaze_points: torch.Tensor,
    gaze_vecs: torch.Tensor,
    inouts: torch.Tensor,
    pids: torch.Tensor,
    gaze_heatmaps: torch.Tensor,
    colors: List,
    heatmap_pid=None,
    frame_nb=None,
    exp_path=None,
    alpha: float = 0.5,
    io_thr: float = 0.7,
    gaze_pt_size: int = 10,
    head_center_size: int = 10,
    thickness: int = 4,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Function to draw gaze results for a single person"""

    # Create canvas on which to draw predictions
    img_height, img_width, _ = image.shape
    canvas = image.copy()

    # Scale of the drawing according to image resolution
    scale = max(img_height, img_width) / 1920
    font_scale *= scale
    thickness = int(scale * thickness)
    gaze_pt_size = int(scale * gaze_pt_size)
    head_center_size = int(scale * head_center_size)
    default_color = [0, 0, 0]

    # Draw heatmap
    if heatmap_pid is not None:
        if len(gaze_heatmaps) == 0:
            raise ValueError(
                "gaze_heatmaps must be provided if heatmap_pid is provided."
            )

        mask = pids == heatmap_pid

        if mask.sum() == 1:  # only if detection found
            gaze_heatmap = gaze_heatmaps[mask]

            heatmap = (
                TF.resize(gaze_heatmap, [img_height, img_width], antialias=True)
                .squeeze()
                .numpy()
            )

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = cm.inferno(heatmap) * 255
            canvas = ((1 - alpha) * image + alpha * heatmap[..., :3]).astype(np.uint8)

            # Write pid being used for the heatmap
            hm_pid_text = f"Heatmap PID: {heatmap_pid}"
            (w_text, h_text) = get_text_size(hm_pid_text, font_scale)

            ul = (img_width - w_text - 20, img_height - h_text - 15)
            br = (img_width, img_height)
            draw_rectangle(canvas, ul, br, default_color, -1)
            hm_pid_text_loc = (img_width - w_text - 10, img_height - 10)
            draw_text(canvas, hm_pid_text, hm_pid_text_loc, font_scale)

    # Draw head bboxes
    if len(head_bboxes) > 0:
        if len(pids) == 0:
            raise ValueError("pids must be provided if head_bboxes is provided")

        # Convert to numpy
        head_bboxes_np = (
            head_bboxes.numpy()
            if isinstance(head_bboxes, torch.Tensor)
            else np.array(head_bboxes)
        )
        if head_bboxes_np.max() <= 1.1:
            head_bboxes_np = head_bboxes_np * np.array(
                [img_width, img_height, img_width, img_height]
            )
        head_bboxes_np = head_bboxes_np.astype(int)

        # Compute head center
        head_centers = np.hstack(
            [
                (head_bboxes_np[:, [0]] + head_bboxes_np[:, [2]]) / 2,
                (head_bboxes_np[:, [1]] + head_bboxes_np[:, [3]]) / 2,
            ]
        )
        head_centers = head_centers.astype(int)

        gaze_available = len(gaze_points) > 0
        if gaze_available and (len(inouts) == 0):
            raise ValueError("inouts must be provided if gaze_pts is provided")
        if gaze_available:
            gaze_points_np = (
                gaze_points.numpy()
                if isinstance(gaze_points, torch.Tensor)
                else np.array(gaze_points)
            )
            if gaze_points.max() <= 1.0:
                gaze_points_np = gaze_points_np * np.array([img_width, img_height])
            gaze_points_np = gaze_points_np.astype(int)

        if gaze_vecs is not None:
            gaze_vecs_np = (
                gaze_vecs.numpy()
                if isinstance(gaze_vecs, torch.Tensor)
                else np.array(gaze_vecs)
            )

        for i, head_bbox in enumerate(head_bboxes_np):
            xmin, ymin, xmax, ymax = head_bbox
            head_radius = max(xmax - xmin, ymax - ymin) // 2
            pid = pids[i]
            color = colors[pid % len(colors)]

            # Compute Head Center
            head_center = head_centers[i]
            draw_circle(canvas, head_center, head_radius, color, thickness)

            # Draw header
            io = inouts[i] if inouts is not None else "-"
            header_text = f"Person {pid}"
            (w_text, h_text) = get_text_size(header_text, font_scale, 1)
            header_ul = (int(head_center[0] - w_text / 2), int(ymin - thickness / 2))
            header_br = (int(head_center[0] + w_text / 2), int(ymin + h_text + 5))
            draw_rectangle(canvas, header_ul, header_br, color, -1)
            draw_text(
                canvas, header_text, (header_ul[0], int(ymin + h_text)), font_scale
            )

            if gaze_available and (io > io_thr):
                gp = gaze_points_np[i]
                vec = gp - head_center
                vec = vec / (np.linalg.norm(vec) + 0.000001)
                intersection = head_center + (vec * head_radius).astype(int)
                draw_line(canvas, intersection, gp, color, thickness)
                draw_circle(canvas, gp, gaze_pt_size, color, -1)

            if gaze_vecs_np is not None:
                gv = gaze_vecs_np[i]
                draw_arrowed_line(
                    canvas,
                    head_center,
                    (head_center + 100 * gv).astype(int),
                    color,
                    thickness,
                )

    # Write frame number
    if frame_nb is not None:
        frame_nb = str(img_width)
        (w_text, h_text) = get_text_size(frame_nb, font_scale, 1)

        nb_ul = (int((img_width - w_text) / 2), (img_height - h_text - 15))
        nb_br = (int((img_width + w_text) / 2), img_height)
        draw_rectangle(canvas, nb_ul, nb_br, default_color, 1)
        nb_text_loc = (int((img_width - w_text) / 2), (img_height - 10))
        draw_text(canvas, frame_nb, nb_text_loc, font_scale)

    # Write experiment name
    if exp_path is not None:
        exp_path = "/".join(exp_path.split("/")[-4:])
        exp_text = f"Experiment: {exp_path}"
        (w_text, h_text) = get_text_size(exp_text, font_scale, 1)
        ul = (0, img_height - h_text - 15)
        br = (w_text + 20, img_height)
        draw_rectangle(canvas, ul, br, default_color, 1)
        exp_text_loc = (10, img_height - 10)
        draw_text(canvas, exp_text, exp_text_loc, font_scale)

    # Draw social gaze
    num_people = len(head_bboxes_np)
    # Iterate over tasks
    indices = torch.tensor(
        list(itertools.permutations(torch.arange(num_people + 1), 2))
    )
    lah_person = np.zeros(num_people + 1, dtype=np.int16) - 1
    laeo_person = np.zeros(num_people + 1, dtype=np.int16) - 1
    coatt_person = []
    for i in range(num_people + 1):
        coatt_person.append([])
    social_persons = [lah_person, laeo_person, coatt_person]

    thresholds = [0.7, 0.7, 0.4]
    for t, task in enumerate(["lah", "laeo", "coatt"]):
        social_pred = social_preds[t]
        social_person = social_persons[t]
        thres = thresholds[t]
        # iterate over pairs
        for pair_num, pair_score in enumerate(social_pred[0]):
            if pair_score > thres:
                pair_indices = indices[pair_num]
                if pair_indices[0] == 0 or pair_indices[1] == 0:
                    continue
                if task == "coatt":
                    social_person[pair_indices[0]].append(pair_indices[1].item())
                    social_person[pair_indices[1]].append(pair_indices[0].item())
                else:
                    if task == "laeo":
                        social_person[pair_indices[0]] = pair_indices[1].item()
                    social_person[pair_indices[1]] = pair_indices[0].item()

    # iterate over tasks
    text_offset = 0
    font_scale = 1.7 * (img_height / 720)
    thickness_text = int(5 * (img_height / 720))
    # iterate over heads
    for i in range(num_people):
        text_offset = 0
        pid = pids[i]
        color = colors[pid % len(colors)]
        io = inouts[i]
        if not io > io_thr:
            continue
        for t, task in enumerate(["lah", "laeo", "coatt"]):
            social_person = social_persons[t]
            # write task over head bbox
            flag = 0
            if task == "coatt":
                if social_person[i + 1] != []:
                    flag = 1
            else:
                if social_person[i + 1] != -1:
                    flag = 1
            if flag:
                head_bbox = head_bboxes_np[i]
                if task == "coatt":
                    text_offset += 46
                    org = (head_bbox[0] + 25, head_bbox[3] + text_offset)
                    out_name = ",".join(
                        [
                            str(pids[z - 1].item())
                            for z in list(set(social_person[i + 1]))
                        ]
                    )
                    draw_text(
                        canvas, "SA " + out_name, org, font_scale, color, thickness_text
                    )
                else:
                    text_offset += 46
                    org = (head_bbox[0] + 25, head_bbox[3] + text_offset)
                    out_text = (
                        task.upper() + " " + str(pids[social_person[i + 1] - 1].item())
                    )
                    draw_text(canvas, out_text, org, font_scale, color, thickness_text)

    return canvas

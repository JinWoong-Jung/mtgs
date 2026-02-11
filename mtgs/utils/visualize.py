# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os

import itertools
import yaml

import cv2
import numpy as np

import torch

import logging

logger = logging.getLogger(__name__)

def get_root(batch):

    with open(os.path.join(os.path.dirname(__file__), "../config/config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    root_coatt = cfg["data"]["videocoatt"]["root"]
    root_vat = cfg["data"]["vat"]["root"]
    root_laeo = cfg["data"]["uco_laeo"]["root"]
    root_childplay = cfg["data"]["childplay"]["root"]
    root = ""

    if "dataset" in batch.keys():
        if batch["dataset"][0] == "videoattentiontarget":
            root = os.path.join(root_vat, "images")
        elif batch["dataset"][0] == "childplay":
            root = root_childplay
        elif batch["dataset"][0] == "laeo":
            root = os.path.join(root_laeo, "images_Idiap")
        elif batch["dataset"][0] == "coatt":
            root = os.path.join(root_coatt, "images_Idiap")

    return root


# auxiliary functions for drawing
colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (0, 255, 165),
    (255, 165, 0),
    (0, 165, 255),
    (255, 0, 165),
    (165, 255, 0),
    (165, 255, 165),
    (255, 165, 165),
    (165, 165, 255),
    (165, 165, 0),
    (165, 85, 85),
    (85, 0, 85),
    (85, 85, 85),
]

# get pixel coordinates for head bbox


def perc_to_pixel(head_bbox, frame_height, frame_width):
    head_bbox[0] *= frame_width
    head_bbox[1] *= frame_height
    head_bbox[2] *= frame_width
    head_bbox[3] *= frame_height

    return head_bbox


# function for visualizing gaze predictions


def visualize_gaze(batch, social_preds):
    root = get_root(batch)

    frame = cv2.imread(os.path.join(root, batch["path"][0]))
    frame_height, frame_width = frame.shape[:2]
    # iterate over all people
    num_people = len(batch["head_bboxes"][0])
    for i in range(1, num_people):
        # draw head bbox
        head_bbox = batch["head_bboxes"][0][i].clone()
        head_bbox = (
            perc_to_pixel(head_bbox, frame_height, frame_width).int().cpu().numpy()
        )
        color = colors[i]
        thickness = 3
        frame = cv2.rectangle(frame, head_bbox[:2], head_bbox[2:], color, thickness)

        # write pid
        org = (head_bbox[0], head_bbox[1] + 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color_text = (255, 255, 255)
        frame = cv2.putText(
            frame, str(i), org, font, fontScale, color_text, thickness, cv2.LINE_AA
        )

        # draw gaze vector
        gaze_vec_pred = batch["gv_pred"][0][i]
        start = (head_bbox[:2] + head_bbox[2:]) / 2
        gv_pred = gaze_vec_pred[0][i].clone().cpu().numpy()
        end = start + gv_pred * 100
        frame = cv2.line(
            frame, start.astype(np.int16), end.astype(np.int16), color_text, thickness
        )

        # draw gaze point
        gaze_pt_pred = batch["gp_pred"][0][i]
        start = (head_bbox[:2] + head_bbox[2:]) / 2
        gp_pred = gaze_pt_pred[0][i].clone().cpu().numpy()
        gp_pred *= [frame_width, frame_height]
        frame = cv2.line(
            frame, start.astype(np.int16), gp_pred.astype(np.int16), color, thickness
        )
        radius = 10
        frame = cv2.circle(frame, gp_pred.astype(np.int16), radius, color, thickness=-1)

    # iterate over tasks
    indices = torch.tensor(list(itertools.permutations(torch.arange(num_people), 2)))
    lah_person = np.zeros(num_people, dtype=np.int16) - 1
    laeo_person = np.zeros(num_people, dtype=np.int16) - 1
    coatt_person = []
    for i in range(num_people):
        coatt_person.append([])
    social_persons = [lah_person, laeo_person, coatt_person]

    thresholds = [0.6, 0.6, 0.6]
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
    for t, task in enumerate(["lah", "laeo", "coatt"]):
        social_person = social_persons[t]
        text_offset += 30
        # iterate over heads
        for i in range(num_people):
            color = colors[i]
            # write task over head bbox
            flag = 0
            if task == "coatt":
                if social_person[i] != []:
                    flag = 1
            else:
                if social_person[i] != -1:
                    flag = 1
            if flag:
                head_bbox = batch["head_bboxes"][0][i].clone()
                head_bbox = (
                    perc_to_pixel(head_bbox, frame_height, frame_width)
                    .int()
                    .cpu()
                    .numpy()
                )
                org = (head_bbox[0], head_bbox[3] + text_offset)
                frame = cv2.putText(
                    frame,
                    task + ": " + str(social_person[i]),
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

    return frame

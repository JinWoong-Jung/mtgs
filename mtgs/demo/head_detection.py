# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
import sys
import numpy as np

import torch

from mtgs.tools.yolov5crowdhuman import attempt_load
from mtgs.utils import check_file, check_folder, get_device
from mtgs.tools.yolov5crowdhuman import (
    non_max_suppression,
    check_img_size,
    scale_coords,
    letterbox,
)

# Add yolo models
yolo_folder = os.path.join(os.getcwd(), "../mtgs/tools/yolov5crowdhuman/")
check_folder(yolo_folder)
sys.path.insert(0, yolo_folder)


class HeadDetector:
    def __init__(
        self,
        checkpoint_file: str,
        device: str,
    ) -> None:
        self.checkpoint_file = checkpoint_file
        self.device = get_device(device)
        self.init_head_detector()

    def init_head_detector(self) -> None:
        """Initialize YOLO detector for head detection"""
        check_file(self.checkpoint_file)
        self.detector = attempt_load(self.checkpoint_file, map_location="cpu")
        self.detector.to(self.device)
        self.detector.eval()  # eval mode

    def preprocess_image(self, image: np.ndarray, image_size) -> torch.Tensor:
        stride = int(self.detector.stride.max())
        image_size = check_img_size(image_size, s=stride)
        image = letterbox(image, image_size, stride=stride)[0]
        image = image.transpose(2, 0, 1)
        image_np = torch.from_numpy(image)
        image_np = image_np.unsqueeze(0).float().to(self.device)
        image_np = image_np / 255.0 if image_np.max() > 1 else image_np
        return image_np

    def detect_heads(
        self,
        image: np.ndarray,
        image_size: int = 640,
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
    ) -> np.ndarray:
        # Pre-process image
        image_input = self.preprocess_image(image, image_size)
        img_w, img_h = image.shape[1], image.shape[0]

        # Inference (predictions)
        with torch.no_grad():
            preds = self.detector(image_input)[0]

        # Apply NMS
        preds = non_max_suppression(preds, conf_thr, iou_thr)[0]
        preds[:, :4] = scale_coords(
            image_input.shape[2:], preds[:, :4], image.shape
        ).round()
        preds = preds.cpu().numpy()

        # Separate head from person detections
        class_names = self.detector.names
        h_mask = preds[:, -1] == class_names.index("head")
        detections = preds[h_mask, :-1]

        # Filter small heads
        detections = np.array(
            [
                det
                for det in detections
                if (det[2] - det[0]) >= 0.02 * img_w
                and (det[3] - det[1]) >= 0.02 * img_h
            ]
        )

        return detections

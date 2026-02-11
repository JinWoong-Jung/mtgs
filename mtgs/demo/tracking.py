# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import numpy as np

from boxmot import OCSORT


class Tracker:
    def __init__(
        self,
        det_threshold: float = 0.2,
        asso_threshold: float = 0.1,
        max_age: int = 300,
        inertia: float = 0.5,
    ) -> None:
        self.det_threshold = det_threshold
        self.asso_threshold = asso_threshold
        self.inertia = inertia
        self.max_age = max_age
        self.init_tracker()

    def init_tracker(self) -> None:
        """Initialize the OCSORT tracker"""
        self.tracker = OCSORT(
            det_thresh=self.det_threshold,
            asso_threshold=self.asso_threshold,
            max_age=self.max_age,
            inertia=self.inertia,
        )

    def reset_tracker(self) -> None:
        """Reset/initialize tracker"""
        self.init_tracker()

    def update(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Update tracking state with input detections"""
        tracks = self.tracker.update(detections, image)
        return tracks

# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import List

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

import logging

logger = logging.getLogger(__name__)


def build_callbacks(
    checkpoint_folder: str,
    use_lr_monitor: bool = False,
    checkpoint_monitor: str = "metric/val/dist",
    checkpoint_mode: str = "min",
) -> List:
    callbacks = []

    # Model Checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename="best",
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=False,
        verbose=True,
    )
    callbacks.append(checkpoint_cb)

    # Learning Rate Monitor
    if use_lr_monitor:
        lr_monitor_cb = LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        )
        callbacks.append(lr_monitor_cb)

    return callbacks

# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import List, Union, Dict

import numpy as np

from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
    ModelCheckpoint,
)

import logging

logger = logging.getLogger(__name__)


def build_callbacks(
    checkpoint_folder: str,
    use_lr_monitor: bool = False,
    use_swa: bool = False,
    swa_params: Union[Dict, None] = None,
) -> List:
    callbacks = []

    # Model Checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_folder,
        filename="best",
        monitor="metric/val/dist",
        mode="min",
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

    # Stochastic Weight Averaging
    if use_swa and swa_params is not None:
        swa_lrs = np.array(swa_params["lr"])

        if len(swa_lrs.shape) > 0:
            swa_lrs = swa_lrs.tolist()
        else:
            swa_lrs = swa_lrs.item()

        logger.info("Using Stochastic Weight Averaging (SWA)")
        swa_callback = StochasticWeightAveraging(
            swa_lrs=swa_lrs,
            swa_epoch_start=swa_params["epoch_start"],
            annealing_epochs=swa_params["annealing_epochs"],
            # device=None # using gpu may overflow the gpu memory
        )
        callbacks.append(swa_callback)

    return callbacks

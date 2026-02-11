# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import List, Union

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.trainer import Trainer


def build_trainer(
    callbacks: Union[List, None],
    train_logger: Union[WandbLogger, bool],
    epochs: int,
    precision,
    device: str = "cuda",
    accumulate_grad_batches: int = 1,
) -> Trainer:
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "auto",
        precision=precision,
        logger=train_logger,
        callbacks=callbacks,
        # uncover bugs without any lengthy training by running all the code.
        # Doesn't generate logs or checkpoints
        fast_dev_run=False,
        max_epochs=epochs,
        # overfit one or a few batches to find bugs. Set it to 0 to disable
        overfit_batches=0.0,
        # int for nb of batches or float in [0., 1.] for fraction of the
        # training epoch
        val_check_interval=1.0,
        # Use None to validate every n batches through
        # `val_check_interval`. default is 1
        check_val_every_n_epoch=1,
        # Sanity check runs n val batches before the training routine.
        # Set to -1 to run all batches
        num_sanity_val_steps=2,
        # If True, enable checkpointing. Configures a default one if
        # there is no ModelCheckpoint callback
        enable_checkpointing=True,
        enable_progress_bar=True,  # Whether to enable to progress bar
        enable_model_summary=True,  # Whether to enable model summarization
        # accumulate gradients every k batches
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=None,  # clip gradients to this value
        gradient_clip_algorithm=None,  # "value" or "norm"
        # guarantee reproducible results by removing most of the
        # randomness from training, but may slow it down
        deterministic=False,
        # set to True to speed up training if the input sizes for
        # your model are fixed (e.g. during inference)
        benchmark=True,
        # Whether to use torch.inference_mode() or torch.no_grad()
        # during evaluation (ie. validate/test/predict)
        inference_mode=False,
        profiler=None,  # None, "simple" or "advanced" to identify bottlenecks
        # Enable anomaly detection for the autograd engine,
        detect_anomaly=False,
    )
    return trainer

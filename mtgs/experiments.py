# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
import torch
import lightning.pytorch as pl

from mtgs.config.config_manager import ConfigManager
from mtgs.networks.models import MTGSModel
from mtgs.utils import create_train_logger
from mtgs.train import (
    build_callbacks,
    build_dataset,
    build_trainer,
)

import logging

logger = logging.getLogger("Experiment")


class Experiment:
    def __init__(self) -> None:
        self.cfg = ConfigManager.get_config()  # Set configuration values

        logger.info(f"Experiment: {self.cfg.experiment.name}")
        logger.info(f"Dataset: {self.cfg.experiment.dataset}")
        logger.info(f"Tasks: {self.cfg.experiment.task}")
        logger.info(f"Output folder: {self.cfg.experiment.output_folder}")

        # Create output folder
        os.makedirs(self.cfg.experiment.output_folder, exist_ok=True)

        # Experiment tasks (eg. train, test, etc)
        self.tasks = self.cfg.experiment.task.split("+")

        # Set manual precision
        torch.set_float32_matmul_precision(self.cfg.train.matmul_precision)

        # Set random seed
        if self.cfg.train.seed is not None:
            pl.seed_everything(self.cfg.train.seed)

        # Set dataset module
        self.data = build_dataset(**self.cfg)

        # Create mtgs model
        self.model = MTGSModel(self.cfg)

        # Create training logger using wandb
        self.train_logger = create_train_logger(self.cfg)

        # Initialize callbacks (for training)
        self.callbacks = None
        if "train" in self.tasks:
            ckpt_folder = os.path.join(
                self.cfg.experiment.output_folder, "train/checkpoints/"
            )
            self.callbacks = build_callbacks(
                checkpoint_folder=ckpt_folder,
                use_lr_monitor=self.cfg.wandb.log,
                use_swa=self.cfg.train.swa.use,
                swa_params={
                    "lr": self.cfg.train.swa.lr,
                    "epoch_start": self.cfg.train.swa.epoch_start,
                    "annealing_epochs": self.cfg.train.swa.annealing_epochs,
                },
                checkpoint_monitor=self.cfg.train.checkpoint_monitor,
                checkpoint_mode=self.cfg.train.checkpoint_mode,
            )

        # Initialize -lighting- trainer
        self.trainer = build_trainer(
            callbacks=self.callbacks,
            train_logger=self.train_logger,
            epochs=self.cfg.train.epochs,
            device=self.cfg.device,
            precision=self.cfg.train.precision,
            accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,
        )

    def train(self) -> None:
        """Train the MTGS model"""

        # Log model parameters and/or gradients (wandb)
        if (
            self.cfg.wandb.log
            and self.cfg.wandb.watch is not None
            and not isinstance(self.train_logger, bool)
        ):
            logger.info(f"Tracking model enabled: {self.cfg.wandb.watch}")
            self.train_logger.watch(
                self.model,
                log=self.cfg.wandb.watch,
                log_freq=self.cfg.wandb.watch_freq,
                log_graph=False,
            )

        # Train
        ckpt = self.cfg.train.resume_from if self.cfg.train.resume else None
        logger.info(f"Resuming model training from: {ckpt}")
        self.trainer.fit(self.model, self.data, ckpt_path=ckpt)

    def validate(self) -> None:
        """Validate the trained MTGS model"""
        ckpt = self.cfg.val.checkpoint
        logger.info(f"Validating model from: {ckpt}")
        self.trainer.validate(self.model, self.data, ckpt_path=ckpt, verbose=True)

    def test(self) -> None:
        """Test the trained MTGS model"""
        ckpt = self.cfg.test.checkpoint if ("train" not in self.tasks) else "best"
        logger.info(f"Testing model from: {ckpt}")

        # When test follows training, advance the internal batch counter by 1 so that
        # test metrics are logged at a strictly higher wandb step than the final
        # training/validation metrics (prevents step collision that hides them in wandb).
        if "train" in self.tasks:
            self.trainer.fit_loop.epoch_loop._batches_that_stepped += 1

        self.trainer.test(self.model, self.data, ckpt_path=ckpt, verbose=True)

    def run(self) -> None:
        """Run experiment (training, testing or validation of MTGS)"""
        logger.info("Starting the experiment")

        # Train MTGS model
        if "train" in self.tasks:
            logger.info("Start the training")
            self.train()

        # Validation is already included in training
        if ("val" in self.tasks) and ("train" not in self.tasks):
            logger.info("Start the validation")
            self.validate()

        # Test the trained MTGS model
        if "test" in self.tasks:
            logger.info("Start the testing")
            self.test()

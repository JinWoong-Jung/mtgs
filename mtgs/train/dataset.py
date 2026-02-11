# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import Dict

from mtgs.utils import Stage
from mtgs.datasets import (
    VideoAttentionTargetDataModule,
    VSGazeDataModule,
    GazeFollowDataModule,
    VideoCoAttDataModule,
    VideoLAEODataModule,
    ChildPlayDataModule,
)

import logging

logger = logging.getLogger(__name__)


def build_dataset(**kwargs):
    """Build the dataset using the configuration file"""

    dataset_name = kwargs["experiment"]["dataset"]
    batch_size_train = kwargs["train"]["batch_size"]
    batch_size_test = kwargs["test"]["batch_size"]
    batch_size_val = kwargs["val"]["batch_size"]
    num_people = kwargs["data"]["num_people"]
    image_size = kwargs["data"]["image_size"]
    head_size = kwargs["model"]["head_size"]
    heatmap_size = kwargs["data"]["heatmap_size"]
    return_head_mask = kwargs["data"]["return_head_mask"]
    temporal_context = kwargs["data"]["temporal_context"]
    temporal_stride = kwargs["data"]["temporal_stride"]
    max_train_samples = kwargs["data"]["max_train_samples"]
    max_val_samples = kwargs["data"]["max_val_samples"]
    max_test_samples = kwargs["data"]["max_test_samples"]

    def _get_batch_size(use_stage: bool) -> Dict:
        if use_stage:
            batch_size = {
                Stage.TRAIN: batch_size_train,
                Stage.VAL: batch_size_val,
                Stage.TEST: batch_size_test,
            }
        else:
            batch_size = {
                "train": batch_size_train,
                "val": batch_size_val,
                "test": batch_size_test,
            }
        return batch_size

    def _get_num_people() -> Dict:
        return {"train": num_people, "val": num_people, "test": "all"}

    # Parameters for the given dataset (not vsgaze)
    dataset_params = {}
    if dataset_name != "vsgaze":
        if dataset_name not in kwargs["data"].keys():
            logger.info(f"Dataset {dataset_name} not defined in config file")
            raise NotImplementedError
        dataset_params = kwargs["data"][dataset_name]

    # Dataset data
    data = None

    if dataset_name == "gazefollow":
        data = GazeFollowDataModule(
            root=dataset_params["root"],
            ann_root=kwargs["data"]["ann_root"],
            batch_size=_get_batch_size(use_stage=False),
            image_size=image_size,
            head_size=head_size,
            heatmap_size=heatmap_size,
            num_people=_get_num_people(),
            return_head_mask=return_head_mask,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    if dataset_name == "childplay":
        data = ChildPlayDataModule(
            root=dataset_params["root"],
            ann_root=kwargs["data"]["ann_root"],
            batch_size=_get_batch_size(use_stage=True),
            image_size=image_size,
            head_size=head_size,
            heatmap_size=heatmap_size,
            num_people=_get_num_people(),
            temporal_context=temporal_context,
            temporal_stride=temporal_stride,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    if dataset_name == "vat":
        data = VideoAttentionTargetDataModule(
            root=dataset_params["root"],
            ann_root=kwargs["data"]["ann_root"],
            batch_size=_get_batch_size(use_stage=True),
            num_people=_get_num_people(),
            temporal_context=temporal_context,
            temporal_stride=temporal_stride,
            image_size=image_size,
            head_size=head_size,
            heatmap_size=heatmap_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    if dataset_name == "videocoatt":
        data = VideoCoAttDataModule(
            root=dataset_params["root"],
            ann_root=kwargs["data"]["ann_root"],
            image_size=image_size,
            head_size=head_size,
            batch_size=_get_batch_size(use_stage=True),
            num_people=_get_num_people(),
            temporal_context=temporal_context,
            temporal_stride=temporal_stride,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    if dataset_name == "uco_laeo":
        data = VideoLAEODataModule(
            root=dataset_params["root"],
            ann_root=kwargs["data"]["ann_root"],
            image_size=image_size,
            head_size=head_size,
            batch_size=_get_batch_size(use_stage=True),
            num_people=_get_num_people(),
            temporal_context=temporal_context,
            temporal_stride=temporal_stride,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    if dataset_name == "vsgaze":
        data = VSGazeDataModule(
            root_coatt=kwargs["data"]["videocoatt"]["root"],
            root_laeo=kwargs["data"]["uco_laeo"]["root"],
            root_vat=kwargs["data"]["vat"]["root"],
            root_childplay=kwargs["data"]["childplay"]["root"],
            ann_root=kwargs["data"]["ann_root"],
            batch_size=_get_batch_size(use_stage=True),
            num_people=_get_num_people(),
            temporal_context=temporal_context,
            temporal_stride=temporal_stride,
            image_size=image_size,
            head_size=head_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    assert data is not None, f"Dataset {dataset_name} not implemented"
    logger.info(f"Dataset: {dataset_name}")

    return data

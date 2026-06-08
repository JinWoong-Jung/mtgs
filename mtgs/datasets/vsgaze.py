# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import Union, Dict, Tuple

import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset, Subset

from mtgs.utils import Stage, pair
from mtgs.train.transforms import (
    RandomCropSafeGaze,
    ColorJitter,
    Normalize,
    ToTensor,
    Compose,
    Resize,
)

from mtgs.datasets.videoattentiontarget_temporal import (
    VideoAttentionTargetDataset_temporal,
)
from mtgs.datasets.videocoatt_temporal import VideoCoAttDataset_temporal
from mtgs.datasets.childplay_temporal import ChildPlayDataset_temporal
from mtgs.datasets.uco_laeo_temporal import VideoLAEODataset_temporal
from mtgs.utils.image import IMG_MEAN, IMG_STD


class VSGazeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_coatt: str,
        root_laeo: str,
        root_vat: str,
        root_childplay: str,
        ann_root: str,
        batch_size: Union[int, dict],
        num_people: Dict[str, int],
        temporal_context: int,
        temporal_stride: int,
        image_size: Tuple[int, int],
        head_size: Tuple[int, int],
        max_train_samples: Union[int, None] = None,
        max_val_samples: Union[int, None] = None,
        max_test_samples: Union[int, None] = None,
    ):
        super().__init__()
        self.root_coatt = root_coatt
        self.root_laeo = root_laeo
        self.root_vat = root_vat
        self.root_childplay = root_childplay
        self.ann_root = ann_root
        self.num_people = num_people
        self.batch_size = (
            {stage: batch_size for stage in Stage}
            if isinstance(batch_size, int)
            else batch_size
        )
        self.temporal_context = temporal_context
        self.temporal_stride = temporal_stride
        self.image_size = pair(image_size)
        self.head_size = head_size
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

    def setup(self, stage: str):
        if stage == "fit":
            ############ Train ##############
            train_transform = Compose(
                [
                    RandomCropSafeGaze(
                        aspect=(self.image_size[0] / self.image_size[1]), p=1.0
                    ),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=self.image_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            dataset_coatt = VideoCoAttDataset_temporal(
                root=self.root_coatt,
                ann_root=self.ann_root,
                split="train",
                stride=max(3, self.temporal_context * self.temporal_stride * 2),
                transform=train_transform,
                tr=(-0.1, 0.1),
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )
            dataset_laeo = VideoLAEODataset_temporal(
                root=self.root_laeo,
                ann_root=self.ann_root,
                split="train",
                stride=max(3, self.temporal_context * self.temporal_stride * 2),
                transform=train_transform,
                tr=(-0.1, 0.1),
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                root=self.root_vat,
                ann_root=self.ann_root,
                split="train",
                stride=max(3, self.temporal_context * self.temporal_stride * 2),
                transform=train_transform,
                tr=(-0.1, 0.1),
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )
            dataset_childplay = ChildPlayDataset_temporal(
                root=self.root_childplay,
                ann_root=self.ann_root,
                split="train",
                stride=max(3, self.temporal_context * self.temporal_stride * 2),
                transform=train_transform,
                tr=(-0.1, 0.1),
                num_people=self.num_people["train"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )

            self.train_dataset = ConcatDataset(
                [dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt]
            )

            ########## val #############
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )

            dataset_coatt = VideoCoAttDataset_temporal(
                root=self.root_coatt,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                head_size=self.head_size,
            )
            dataset_laeo = VideoLAEODataset_temporal(
                root=self.root_laeo,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                head_size=self.head_size,
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                root=self.root_vat,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                head_size=self.head_size,
            )
            dataset_childplay = ChildPlayDataset_temporal(
                root=self.root_childplay,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                head_size=self.head_size,
            )

            self.val_dataset = ConcatDataset(
                [dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt]
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )

            dataset_coatt = VideoCoAttDataset_temporal(
                root=self.root_coatt,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )
            dataset_laeo = VideoLAEODataset_temporal(
                root=self.root_laeo,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                root=self.root_vat,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )
            dataset_childplay = ChildPlayDataset_temporal(
                root=self.root_childplay,
                ann_root=self.ann_root,
                split="val",
                stride=6,
                transform=val_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["val"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
            )

            self.val_dataset = ConcatDataset(
                [dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt]
            )

        elif stage == "test":
            aspect = False  # maintain aspect ratio
            if aspect:
                img_size = self.image_size[1]
            else:
                img_size = self.image_size
            test_transform = Compose(
                [
                    Resize(img_size=img_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            dataset_coatt = VideoCoAttDataset_temporal(
                root=self.root_coatt,
                ann_root=self.ann_root,
                split="test",
                stride=1,
                transform=test_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect=aspect,
            )
            dataset_laeo = VideoLAEODataset_temporal(
                root=self.root_laeo,
                ann_root=self.ann_root,
                split="test",
                stride=1,
                transform=test_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect=aspect,
            )
            dataset_vat = VideoAttentionTargetDataset_temporal(
                root=self.root_vat,
                ann_root=self.ann_root,
                split="test",
                stride=1,
                transform=test_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect=aspect,
            )
            dataset_childplay = ChildPlayDataset_temporal(
                root=self.root_childplay,
                ann_root=self.ann_root,
                split="test",
                stride=1,
                transform=test_transform,
                tr=(0.0, 0.0),
                num_people=self.num_people["test"],
                temporal_context=self.temporal_context,
                temporal_stride=self.temporal_stride,
                image_size=self.image_size,
                aspect=aspect,
            )

            self.test_dataset = ConcatDataset(
                [dataset_childplay, dataset_vat, dataset_laeo, dataset_coatt]
            )

    def train_dataloader(self):
        # Use the full dataset or a subset (eg. quick experiments)
        train_dataset = self.train_dataset
        if self.max_train_samples is not None:
            train_dataset = Subset(train_dataset, range(self.max_train_samples))

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size[Stage.TRAIN],
            shuffle=True,
            num_workers=14,
            pin_memory=True,
            persistent_workers=True,
        )
        return dataloader

    def val_dataloader(self):
        # Use the full dataset or a subset (eg. quick experiments)
        val_dataset = self.val_dataset
        if self.max_val_samples is not None:
            val_dataset = Subset(val_dataset, range(self.max_val_samples))

        dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size[Stage.VAL],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        )
        return dataloader

    def test_dataloader(self):
        # Use the full dataset or a subset (eg. quick experiments)
        test_dataset = self.test_dataset
        if self.max_test_samples is not None:
            test_dataset = Subset(test_dataset, range(self.max_test_samples))

        dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size[Stage.TEST],
            shuffle=True,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        )
        return dataloader

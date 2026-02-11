# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
from typing import Dict, Union, Tuple

from omegaconf import OmegaConf, ListConfig

import numpy as np
import pandas as pd
from PIL import Image

import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from mtgs.utils.utils import square_bbox
from mtgs.utils.image import IMG_MEAN, IMG_STD
from mtgs.utils.social_gaze import get_lah_labels, get_shuffle_idx
from mtgs.utils import (
    generate_gaze_heatmap,
    generate_mask,
    pair,
)
from mtgs.train.transforms import (
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    RandomCropSafeGaze,
    ColorJitter,
    Normalize,
    ToTensor,
    Compose,
    Resize,
)


class GazeFollowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        ann_root: str,
        batch_size: Union[int, dict],
        image_size: Tuple[int, int],
        head_size: Tuple[int, int],
        heatmap_size: int,
        num_people: dict = {"train": 1, "val": 1, "test": 1},
        return_head_mask: bool = False,
        max_train_samples: Union[int, None] = None,
        max_val_samples: Union[int, None] = None,
        max_test_samples: Union[int, None] = None,
    ):
        super().__init__()
        self.root = root
        self.ann_root = ann_root
        if type(image_size) == ListConfig:
            image_size = OmegaConf.to_object(image_size)
        self.image_size = pair(image_size)
        self.head_size = head_size
        self.heatmap_sigma = int(np.mean(heatmap_size) * 3 / 64)
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.batch_size = (
            {stage: batch_size for stage in ["train", "val", "test"]}
            if isinstance(batch_size, int)
            else batch_size
        )
        self.return_head_mask = return_head_mask
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(
                        aspect=(self.image_size[0] / self.image_size[1]), p=1
                    ),
                    RandomHorizontalFlip(p=0.5),
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
            self.train_dataset = GazeFollowDataset(
                self.root,
                self.ann_root,
                "train",
                train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["train"],
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.ann_root,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["val"],
                return_head_mask=self.return_head_mask,
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.ann_root,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["val"],
                return_head_mask=self.return_head_mask,
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    # maintain aspect ratio while testing
                    Resize(img_size=self.image_size, head_size=self.head_size),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = GazeFollowDataset(
                self.root,
                self.ann_root,
                "test",
                test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["test"],
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        # Use the full dataset or a subset (eg. quick experiments)
        train_dataset = self.train_dataset
        if self.max_train_samples is not None:
            train_dataset = Subset(train_dataset, range(self.max_train_samples))

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=14,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        # Use the full dataset or a subset (eg. quick experiments)
        val_dataset = self.val_dataset
        if self.max_val_samples is not None:
            val_dataset = Subset(val_dataset, range(self.max_val_samples))

        dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        # Use the full dataset or a subset (eg. quick experiments)
        test_dataset = self.test_dataset
        if self.max_test_samples is not None:
            test_dataset = Subset(test_dataset, range(self.max_test_samples))

        dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader


# ============================================================================= #
#                               GAZEFOLLOW DATASET                              #
# ============================================================================= #
class GazeFollowDataset(Dataset):
    def __init__(
        self,
        root,
        ann_root,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 5,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), (
            f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        )
        assert (num_people == "all") or (num_people > 0), (
            f'Expected `num_people` to be strictly positive or "all", but received {num_people} instead.'
        )
        assert 0 <= head_thr <= 1, (
            f"Expected `head_thr` to be in [0, 1]. Received {head_thr} instead."
        )

        self.root = root
        self.ann_root = ann_root
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask

        # load annotations
        self.annotations = pd.read_hdf(
            os.path.join(ann_root, f"gazefollow_{self.split}.h5"), "data"
        )
        self.annotations = self.annotations.groupby("path")
        self.paths = list(self.annotations.groups.keys())
        self.length = len(self.paths)

    def __getitem__(self, index: int) -> Dict:
        # load annotations
        path = self.paths[index]
        img_annotations = self.annotations.get_group(path).iloc[0]
        pids = torch.from_numpy(img_annotations["person_ids"])
        inout = torch.from_numpy(img_annotations["inout"].astype(np.float32))

        if self.split != "test":
            gaze_pts = torch.from_numpy(
                img_annotations["gaze_points"].astype(np.float32)
            )
        else:
            gaze_pts = torch.from_numpy(
                img_annotations["gaze_points_p1"].astype(np.float32)
            )

        # Load image
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = image.size

        # Load head bboxes
        head_bboxes = img_annotations["head_bboxes"]
        head_bboxes = torch.from_numpy(head_bboxes.astype(np.float32))
        head_bboxes = head_bboxes * torch.tensor([img_w, img_h, img_w, img_h])
        # # expand bbox slightly
        # head_bboxes = expand_bbox(head_bboxes, img_w, img_h, k=0.1)

        # Shuffle people
        if self.split == "train" and len(head_bboxes) > 2:
            rand_indices = get_shuffle_idx(inout.numpy())
            pids = pids[rand_indices]
            head_bboxes = head_bboxes[rand_indices]
            gaze_pts = gaze_pts[rand_indices]
            inout = inout[rand_indices]

        # Jitter head bboxes
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract Heads (negative values add padding)
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))  # type:ignore

        # Select {1, num_people} people
        num_heads = len(heads)
        num_keep = num_heads
        if self.num_people != "all":
            if num_heads > 1:
                num_keep = np.random.randint(
                    2, min(num_heads, self.num_people) + 1
                )  # min 2 people
        #                 num_keep = np.random.randint(1, min(num_heads, self.num_people)+1)   # min one person
        head_bboxes = head_bboxes[-num_keep:]
        heads = heads[-num_keep:]
        pids = pids[-num_keep:]
        if self.split != "test":
            gaze_pts = gaze_pts[-num_keep:]
            inout = inout[-num_keep:]
        num_heads = len(heads)
        # pad at least one person
        num_missing_heads = (
            max(self.num_people + 1 - num_heads, 1) if self.num_people != "all" else 1
        )

        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt); always have one extra person
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4)), head_bboxes])
            heads = (
                num_missing_heads
                * [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))]
                + heads
            )
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2)) - 1, gaze_pts])
            pids = torch.cat([torch.zeros(num_missing_heads) - 1, pids])
        if self.split != "test":
            inout = torch.cat(
                [torch.zeros((num_missing_heads,), dtype=torch.float32) - 1, inout]
            )

        # Normalize Head Bboxes
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

        # Get LAH labels
        pairs_all = img_annotations["pairs"]
        lah_labels_all = img_annotations["lah_pairs"]
        lah_labels = get_lah_labels(pids, pairs_all, lah_labels_all)

        # Build Sample
        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pts": gaze_pts,
            "inout": inout,
            "coatt_labels": torch.zeros_like(lah_labels) - 1,
            "lah_labels": lah_labels,
            "laeo_labels": torch.zeros_like(lah_labels) - 1,
            "num_valid_people": torch.tensor(num_heads),
            "is_child": torch.zeros(len(heads), dtype=torch.float) - 1,
            "speaking": torch.zeros(len(heads), dtype=torch.float) - 1,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": [path],
            "dataset": "gazefollow",
        }

        # Transform
        if self.transform:
            sample = self.transform(sample)

        # compute head masks
        _, img_h, img_w = sample["image"].shape
        sample["head_masks"] = generate_mask(sample["head_bboxes"], img_w, img_h)

        # Compute Head Bbox Centers
        head_bboxes = sample["head_bboxes"]
        gaze_pts = sample["gaze_pts"]
        head_centers = torch.hstack(
            [
                (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2,
                (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2,
            ]
        )
        sample["head_centers"] = head_centers

        # compute 2d gaze vectors
        if self.split == "test":
            # only consider annotated person
            head_centers = head_centers[-1].unsqueeze(0)
        sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)

        if self.split != "test":
            # generate gaze heatmaps
            sample["gaze_heatmaps"] = generate_gaze_heatmap(
                sample["gaze_pts"], sigma=self.heatmap_sigma, size=self.heatmap_size
            )

        # add extra dimension to be compatible with temporal model
        for key, item in sample.items():
            if key not in ["path", "dataset"]:
                sample[key] = item.unsqueeze(0)

        return sample

    def __len__(self):
        return self.length

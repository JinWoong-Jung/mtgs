# mtgs/datasets/vlm_datamodule.py
"""Stage B datamodule: VSGaze datasets with no random augmentation (val_transform only)."""
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset

from mtgs.datasets.videoattentiontarget_temporal import VideoAttentionTargetDataset_temporal
from mtgs.datasets.childplay_temporal import ChildPlayDataset_temporal
from mtgs.datasets.uco_laeo_temporal import VideoLAEODataset_temporal
from mtgs.datasets.videocoatt_temporal import VideoCoAttDataset_temporal
from mtgs.train.transforms import Resize, ToTensor, Normalize, Compose
from mtgs.train.collate import pad_collate_fn
from mtgs.utils.image import IMG_MEAN, IMG_STD


class VLMDataModule(pl.LightningDataModule):
    """VSGaze datasets with deterministic transform for all splits."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _make_transform(self):
        # Pass a (W, H) tuple so Resize forces a SQUARE image (e.g. 448x448).
        # A bare int makes Resize aspect-preserving (e.g. 448x800), which then
        # mismatches the square zero-padding used for missing frames -> stack error.
        img_sz = self.cfg.data.image_size
        if isinstance(img_sz, int):
            img_sz = (img_sz, img_sz)
        return Compose([
            Resize(img_size=img_sz, head_size=self.cfg.model.head_size),
            ToTensor(),
            Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
        ])

    def _make_dataset(self, split, num_people=None):
        cfg = self.cfg
        t = self._make_transform()
        stride = max(3, cfg.data.temporal_context * cfg.data.temporal_stride * 2)
        img_sz = cfg.data.image_size
        if isinstance(img_sz, int):
            img_sz = (img_sz, img_sz)
        kw = dict(
            split=split,
            stride=stride,
            transform=t,
            tr=(0.0, 0.0),
            num_people=cfg.data.num_people if num_people is None else num_people,
            temporal_context=cfg.data.temporal_context,
            temporal_stride=cfg.data.temporal_stride,
            image_size=img_sz,
        )
        datasets = []
        if cfg.data.get("vat") and cfg.data.vat.get("root"):
            datasets.append(VideoAttentionTargetDataset_temporal(
                root=cfg.data.vat.root, ann_root=cfg.data.ann_root, **kw))
        if cfg.data.get("childplay") and cfg.data.childplay.get("root"):
            datasets.append(ChildPlayDataset_temporal(
                root=cfg.data.childplay.root, ann_root=cfg.data.ann_root, **kw))
        if cfg.data.get("uco_laeo") and cfg.data.uco_laeo.get("root"):
            datasets.append(VideoLAEODataset_temporal(
                root=cfg.data.uco_laeo.root, ann_root=cfg.data.ann_root, **kw))
        if cfg.data.get("videocoatt") and cfg.data.videocoatt.get("root"):
            datasets.append(VideoCoAttDataset_temporal(
                root=cfg.data.videocoatt.root, ann_root=cfg.data.ann_root, **kw))
        return ConcatDataset(datasets)

    def setup(self, stage=None):
        self.train_ds = self._make_dataset("train")
        self.val_ds   = self._make_dataset("val")

    def train_dataloader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.val.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=True,
        )

    def make_test_loader(self):
        """Held-out test split at N=all (every person), batch_size=1.

        Built on demand (independent of setup()) for online evaluation.
        """
        num_workers = getattr(self.cfg.train, "num_workers", 4)
        test_ds = self._make_dataset("test", num_people="all")
        return DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=True,
        )

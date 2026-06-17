# scripts/main_vlm.py
"""Stage B entry point — separate from main.py to avoid any interference."""
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from mtgs.config import ConfigManager
from mtgs.networks.vlm.vlm_trainer import VLMReasonerModel
from mtgs.datasets.vlm_datamodule import VLMDataModule
from mtgs.datasets.cached_vlm_dataset import CachedVLMDataModule
from mtgs.utils.logger import create_train_logger


@hydra.main(
    config_path="./../mtgs/config/",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)
    pl.seed_everything(cfg.train.get("seed", 42))
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    use_cache = bool(cfg.get("vlm", {}).get("feature_cache", {}).get("use", False))
    model = VLMReasonerModel(cfg)

    stage_a_ckpt = cfg.vlm.get("stage_a_ckpt", None)
    if not use_cache:
        if not stage_a_ckpt:
            raise ValueError("vlm.stage_a_ckpt must be set for online MTGS feature extraction")
        model.load_stage_a_weights(stage_a_ckpt)

    datamodule = CachedVLMDataModule(cfg) if use_cache else VLMDataModule(cfg)

    logger = create_train_logger(cfg)
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.experiment.output_folder}/train/checkpoints",
        filename="best",
        monitor="metric/val/social_ap",
        mode="max",
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=False,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=50,
        val_check_interval=1.0,
        default_root_dir=cfg.experiment.output_folder,
        logger=logger,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, datamodule=datamodule)

    # ── Final N=all test pass → LAH/LAEO/SA AP to wandb ────────────────────────
    # Bump the step counter so test metrics log at a strictly higher wandb step
    # than the last train/val metrics (prevents step collision hiding them).
    try:
        trainer.fit_loop.epoch_loop._batches_that_stepped += 1
    except Exception:
        pass
    test_loader = datamodule.make_test_loader()
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()

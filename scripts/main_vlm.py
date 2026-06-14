# scripts/main_vlm.py
"""Stage B entry point — separate from main.py to avoid any interference."""
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import torch

from mtgs.config import ConfigManager
from mtgs.networks.vlm.vlm_trainer import VLMReasonerModel
from mtgs.datasets.vlm_datamodule import VLMDataModule
from mtgs.datasets.cached_vlm_dataset import CachedVLMDataModule


@hydra.main(
    config_path="./../mtgs/config/",
    config_name="config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)
    pl.seed_everything(cfg.train.get("seed", 42))
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)

    model = VLMReasonerModel(cfg)

    use_cache = bool(cfg.get("vlm", {}).get("feature_cache", {}).get("use", False))
    if use_cache:
        # Cached features already encode Stage A — no backbone/ckpt to load.
        datamodule = CachedVLMDataModule(cfg)
    else:
        # Online mode: load Stage A weights into the frozen MTGS backbone.
        stage_a_ckpt = cfg.vlm.get("stage_a_ckpt", None)
        if stage_a_ckpt:
            model.load_stage_a_weights(stage_a_ckpt)
        else:
            raise ValueError("vlm.stage_a_ckpt must be set for Stage B training")
        datamodule = VLMDataModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=50,
        val_check_interval=1.0,
        default_root_dir=cfg.experiment.output_folder,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

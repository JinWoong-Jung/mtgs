# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import hydra
from omegaconf import DictConfig

from mtgs.config import ConfigManager
from mtgs.demo.processor import DemoProcessor


@hydra.main(config_path="../mtgs/config/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)

    # Run MTGS on a video
    processor = DemoProcessor()
    processor.process_video(
        video_file=cfg.demo.video_file,
        output_folder=cfg.demo.output_folder,
    )


if __name__ == "__main__":
    main()

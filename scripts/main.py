# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import hydra
from omegaconf import DictConfig

from mtgs.config import ConfigManager
from mtgs.experiments import Experiment


@hydra.main(
    config_path="./../mtgs/config/", config_name="config.yaml", version_base=None
)
def main(cfg: DictConfig):
    ConfigManager.set_config(cfg)

    # Run experiment (train/test)
    experiment = Experiment()
    experiment.run()


if __name__ == "__main__":
    main()

# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only
# coding=utf-8

from omegaconf import DictConfig


class ConfigManager:
    _cfg: DictConfig = None

    @classmethod
    def set_config(cls, cfg: DictConfig):
        cls._cfg = cfg

    @classmethod
    def get_config(cls) -> DictConfig:
        if cls._cfg is None:
            raise RuntimeError("Config not initialized. Call set_config() first.")
        return cls._cfg

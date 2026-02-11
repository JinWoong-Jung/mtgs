# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import Union

import torch


def get_device(name: Union[str, None] = None):
    assert name in ["gpu", "cuda", "cpu", None], f"Unknown device: {name}"
    if name == "cpu":
        device = torch.device("cpu")
    elif name in ["cuda", "gpu"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

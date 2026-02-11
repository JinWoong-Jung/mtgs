# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import torch

from mtgs.utils import check_file, get_device
from mtgs.networks.mtgs_net import MTGS


class GazePredictor:
    def __init__(
        self,
        checkpoint_file: str,
        temporal_context: int,
        image_size: int,
        patch_size: int,
        decoder_feature_dim: int,
        decoder_use_bn: bool,
        device: str,
    ) -> None:
        self.checkpoint_file = checkpoint_file
        self.temporal_context = temporal_context
        self.image_size = image_size
        self.patch_size = patch_size
        self.decoder_feature_dim = decoder_feature_dim
        self.decoder_use_bn = decoder_use_bn
        self.device = get_device(device)
        self.init_gaze_predictor()

    def init_gaze_predictor(self) -> None:
        self.predictor = MTGS(
            image_size=self.image_size,
            patch_size=self.patch_size,
            decoder_feature_dim=self.decoder_feature_dim,
            decoder_use_bn=self.decoder_use_bn,
            temporal_context=self.temporal_context,
        )

        # Load checkpoint
        check_file(self.checkpoint_file)
        checkpoint = torch.load(self.checkpoint_file, map_location="cpu")
        checkpoint = {
            name.replace("model.", ""): value
            for name, value in checkpoint["state_dict"].items()
        }
        self.predictor.load_state_dict(checkpoint, strict=True)
        self.predictor.to(self.device)
        self.predictor.eval()  # eval mode

        # Cleanup
        del checkpoint

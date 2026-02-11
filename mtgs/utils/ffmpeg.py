# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import shlex


def get_video_creation_command(file: str, img_width: int, img_height: int, fps: float):
    command = f"ffmpeg -loglevel error -y -s {img_width}x{img_height} -pixel_format rgb24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {file}"
    command = shlex.split(command)
    return command

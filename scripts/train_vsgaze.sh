#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH -A YOUR_PROJECT
#SBATCH -t 16:00:00
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p gpu
#SBATCH --gpus h100:1
#SBATCH --job-name=vsgaze_train
#SBATCH --output=logs/vsgaze_train_%j.out
#SBATCH --error=logs/vsgaze_train_%j.err

# set arguments
TASKS="train+test"
WEIGHTS="..." # path to the checkpoint you want to start from

python ./main.py experiment.task=$TASKS \
    model.weights=$WEIGHTS
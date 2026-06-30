#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vsgaze_test
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/vg_test_%j.out
#SBATCH --error=logs/vg_test_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

EXP_DIR="/home/jinwoongjung/MTGS/experiments/2026-06-28/V14.5-testACF+edit_scheduler"
CHECKPOINT="${EXP_DIR}/train/checkpoints/best.ckpt"

LAEO_DERIVE="decoder"
FROZEN=false

python -s ./main.py experiment.task=test \
    model.weights=False \
    gaze_graph.laeo_derive=$LAEO_DERIVE \
    gaze_graph.frozen=$FROZEN \
    "test.checkpoint='${CHECKPOINT}'" \
    "hydra.run.dir='${EXP_DIR}'"

#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vat_test_batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -p gpu
#SBATCH --output=logs/vat_test_%j.out
#SBATCH --error=logs/vat_test_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR/scripts"

# set arguments
NAME='VAT_test'

TASKS="test"
DATASET='vat'

TEMPORAL_CONTEXT=2 # remember to set temporal context to 0 for static models

TEST_CHECKPOINT="/home/jinwoongjung/MTGS/experiments/2026-05-18/MTGS-dinov2-vitb14-448-VSGaze/train/checkpoints/best.ckpt" # path to the checkpoint you want to test

python main.py experiment.task=$TASKS \
    experiment.name=$NAME \
    experiment.dataset=$DATASET \
    data.temporal_context=$TEMPORAL_CONTEXT \
    test.checkpoint=$TEST_CHECKPOINT 
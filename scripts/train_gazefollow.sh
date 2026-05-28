#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH -A YOUR_PROJECT
#SBATCH -t 08:00:00
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -p gpu
#SBATCH --gpus h100:1
#SBATCH --job-name=gazefollow_train
#SBATCH --output=logs/gazefollow_train_%j.out
#SBATCH --error=logs/gazefollow_train_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR/scripts"

# set arguments
TASKS="train+test"

DATASET="gazefollow"
NUM_SAMPLES=108955

TEMPORAL_CONTEXT=0

BATCH_SIZE=48
OPTIMIZER_LR=1e-4
TO_EPOCHS=4

EPOCHS=20
ANNEALING_EPOCHS=8
ANNEALING_START=11
FINAL_LR=3e-5

GAZE_WEIGHTS="..." # path to the gaze backbone checkpoint

python -s ./main.py experiment.task=$TASKS \
        model.gaze_weights=$GAZE_WEIGHTS \
        experiment.dataset=$DATASET \
        data.num_samples=$NUM_SAMPLES \
        data.temporal_context=$TEMPORAL_CONTEXT \
        optimizer.lr=$OPTIMIZER_LR \
        scheduler.t_0_epochs=$TO_EPOCHS \
        train.epochs=$EPOCHS \
        train.swa.annealing_epochs=$ANNEALING_EPOCHS \
        train.swa.epoch_start=$ANNEALING_START \
        train.swa.lr=$FINAL_LR \
        train.batch_size=$BATCH_SIZE \
        val.batch_size=$BATCH_SIZE
#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=gazefollow_test
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=08:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/gazefollow_graph_test_%j.out
#SBATCH --error=logs/gazefollow_graph_test_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

cd "$SLURM_SUBMIT_DIR/scripts"

# Most recent GazeFollow gaze_graph checkpoint (the run that produced
# test/lah_auc=0.59 in gazefollow_graph_2042.out). Re-test with the
# invalid-edge masking fix in mtgs_net.py applied.
CHECKPOINT="/home/jinwoongjung/MTGS/experiments/2026-06-05/GazeFollow_gaze_graph/train/checkpoints/best.ckpt"

EXP_NAME="test_GazeFollow_gaze_graph"

DATASET="gazefollow"
TEMPORAL_CONTEXT=0

python -s ./main.py experiment.task=test \
        experiment.name=$EXP_NAME \
        model.weights=False \
        experiment.dataset=$DATASET \
        data.temporal_context=$TEMPORAL_CONTEXT \
        test.checkpoint=$CHECKPOINT \
        "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"

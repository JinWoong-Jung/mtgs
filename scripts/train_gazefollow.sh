#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=gf_gaze_graph
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/gf_gaze_graph_%j.out
#SBATCH --error=logs/gf_gaze_graph_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

cd "$SLURM_SUBMIT_DIR/scripts"

EXP_NAME="gf_gaze_graph_fixed"

# ── Dataset / training settings ───────────────────────────────────────────────
TASKS="train+test"
DATASET="gazefollow"
NUM_SAMPLES=108955
TEMPORAL_CONTEXT=0
BATCH_SIZE=48
OPTIMIZER_LR=1e-4
TO_EPOCHS=4
EPOCHS=20

python -s ./main.py experiment.task=$TASKS \
        experiment.name=$EXP_NAME \
        model.weights=False \
        experiment.dataset=$DATASET \
        data.num_samples=$NUM_SAMPLES \
        data.temporal_context=$TEMPORAL_CONTEXT \
        optimizer.lr=$OPTIMIZER_LR \
        scheduler.t_0_epochs=$TO_EPOCHS \
        train.epochs=$EPOCHS \
        train.batch_size=$BATCH_SIZE \
        val.batch_size=$BATCH_SIZE \
        "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"

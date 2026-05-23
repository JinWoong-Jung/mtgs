#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=gazefollow_train
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --output=logs/gazefollow_train_%j.out
#SBATCH --error=logs/gazefollow_train_%j.err

# conda 환경 활성화 (user site-packages 무시하여 ~/.local 충돌 방지)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

# sbatch 제출 디렉토리 기준으로 scripts/ 로 이동
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

GAZE_WEIGHTS="/home/jinwoongjung/MTGS/weights/gaze360_resnet18.pt"

INTERACTION_TYPE="transformer" # "graph" or "transformer"
EXP_NAME="GazeFollow_Training_Type:${INTERACTION_TYPE}"

python ./main.py experiment.task=$TASKS \
        model.gaze_weights=$GAZE_WEIGHTS \
        interaction.type=$INTERACTION_TYPE \
        experiment.name=$EXP_NAME \
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
        val.batch_size=$BATCH_SIZE \
        "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"
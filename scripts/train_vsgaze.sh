#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vsgaze_train
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=96G
#SBATCH -p gpu
#SBATCH --output=logs/vsgaze_train_graph_%j.out
#SBATCH --error=logs/vsgaze_train_graph_%j.err

# conda 환경 활성화 (user site-packages 무시하여 ~/.local 충돌 방지)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

# sbatch 제출 디렉토리 기준으로 scripts/ 로 이동
cd "$SLURM_SUBMIT_DIR/scripts"

# set arguments
TASKS="train+test"
WEIGHTS="/home/jinwoongjung/MTGS/experiments/GazeFollow_Training/train/checkpoints/best.ckpt" # path to the checkpoint you want to start from

INTERACTION_TYPE="graph" # "graph" or "transformer"
DATASET="vsgaze"
EXP_NAME="VSGaze_${INTERACTION_TYPE}"

python ./main.py experiment.task=$TASKS \
    model.weights=$WEIGHTS \
    experiment.name=$EXP_NAME \
    interaction.type=$INTERACTION_TYPE \
    experiment.dataset=$DATASET \
    "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"
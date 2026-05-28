#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vsgaze_train
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/vsgaze_train_graph_%j.out
#SBATCH --error=logs/vsgaze_train_graph_%j.err

# conda 환경 활성화 (user site-packages 무시하여 ~/.local 충돌 방지)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

# sbatch 제출 디렉토리 기준으로 scripts/ 로 이동
cd "$SLURM_SUBMIT_DIR"

# set arguments
TASKS="train+test"
WEIGHTS="/home/jinwoongjung/MTGS/weights/mtgs-static-gazefollow.ckpt" # HuggingFace pretrained GazeFollow checkpoint

INTERACTION_TYPE="graph" # "graph" or "transformer"
EXP_NAME="VSGaze_${INTERACTION_TYPE}_try"

# graph mode has 5 param_groups (vs transformer's 4), so swa.lr must have 5 values
if [ "$INTERACTION_TYPE" = "graph" ]; then
    SWA_LR_OVERRIDE="train.swa.lr=[1e-6,1e-6,1e-6,1e-6,3e-7]"
else
    SWA_LR_OVERRIDE=""
fi

python -s ./main.py experiment.task=$TASKS \
    model.weights=$WEIGHTS \
    experiment.name=$EXP_NAME \
    interaction.type=$INTERACTION_TYPE \
    ${SWA_LR_OVERRIDE} \
    "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"
    
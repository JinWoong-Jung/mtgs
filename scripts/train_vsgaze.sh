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
#SBATCH --output=logs/vg_gaze_graph_%j.out
#SBATCH --error=logs/vg_gaze_graph_%j.err

# conda 환경 활성화 (user site-packages 무시하여 ~/.local 충돌 방지)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

# sbatch를 MTGS/scripts에서 실행하면 그대로 사용하고,
# MTGS 루트에서 실행한 경우에는 scripts/로 이동한다.
if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

# set arguments
TASKS="train+test"
WEIGHTS="/home/jinwoongjung/MTGS/weights/mtgs-static-gazefollow.ckpt"  # GazeFollow stage-1 gaze_graph ckpt

FROZEN=false

EXP_NAME="vg_gaze_graph_v4"
SWA="False"

CHECKPOINT_MONITOR="metric/val/social_ap"
CHECKPOINT_MODE="max"

LAEO_DERIVE="lah_min"

python -s ./main.py experiment.task=$TASKS \
    model.weights=$WEIGHTS \
    "experiment.name='${EXP_NAME}'" \
    gaze_graph.laeo_derive=$LAEO_DERIVE \
    gaze_graph.frozen=$FROZEN \
    train.checkpoint_monitor=$CHECKPOINT_MONITOR \
    train.checkpoint_mode=$CHECKPOINT_MODE \
    train.swa.use=$SWA \
    "hydra.run.dir='\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}'"
    
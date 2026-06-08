#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=postgraph
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/postgraph_%j.out
#SBATCH --error=logs/postgraph_%j.err

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

# ── Post-training: frozen transformer trunk as a visual extractor ─────────────
# 원본 transformer 모드 GazeFollow→VSGaze 완성본(mtgs-vsgaze.ckpt)을 그대로 로드해
# trunk(visual + ViT-Adaptor + people_interaction/temporal + heatmap/inout/gaze
# decoder) 전체를 FREEZE하고, 그 위에 gaze_graph_block만 학습한다.
# gaze_graph 모드는 transformer interaction 모듈을 구조적으로 공유하므로
# strict=False 로드 시 trunk 전부가 채워지고 gaze_graph_block만 random init된다.
TASKS="train+test"
WEIGHTS="/home/jinwoongjung/MTGS/weights/mtgs-vsgaze.ckpt"

INTERACTION_TYPE="gaze_graph"     # 고정: post-training은 gaze_graph 전용
INTERACTION_ORDER="inject_first"  # "inject_first" (original) | "extract_first"

CHECKPOINTS_MONITOR="loss/val/social"
CHECKPOINTS_MODE="min"

EXP_NAME="postgraph"

python -s ./main.py experiment.task=$TASKS \
    model.weights=$WEIGHTS \
    "experiment.name='${EXP_NAME}'" \
    interaction.type=$INTERACTION_TYPE \
    interaction.order=$INTERACTION_ORDER \
    train.freeze.all_but_gaze_graph=true \
    train.checkpoint_monitor="$CHECKPOINTS_MONITOR" \
    train.checkpoint_mode="$CHECKPOINTS_MODE" \
    "hydra.run.dir='\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}'"

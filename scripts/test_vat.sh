#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vat_test
#SBATCH --gres=gpu:mig48gb:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=48G
#SBATCH --output=logs/vat_test_%j.out
#SBATCH --error=logs/vat_test_%j.err

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
NAME='VAT_test'

TASKS="test"
DATASET='vat'

TEMPORAL_CONTEXT=2 # remember to set temporal context to 0 for static models

LAEO_DERIVE="decoder"

TEST_CHECKPOINT="/home/jinwoongjung/MTGS/experiments/MTGS+graph/train/checkpoints/best.ckpt" # path to the checkpoint you want to test

# model.weights=False skips the warm-start load (test.checkpoint overrides all weights anyway).
python -s ./main.py experiment.task=$TASKS \
    experiment.name=$NAME \
    experiment.dataset=$DATASET \
    data.temporal_context=$TEMPORAL_CONTEXT \
    model.weights=False \
    gaze_graph.laeo_derive=$LAEO_DERIVE \
    "test.checkpoint='${TEST_CHECKPOINT}'" \
    "hydra.run.dir='\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${NAME}'"

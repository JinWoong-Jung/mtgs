#!/bin/bash

#SBATCH --job-name=vsgaze_test
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --output=logs/vsgaze_test_transformer_%j.out
#SBATCH --error=logs/vsgaze_test_transformer_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

# sbatch 제출 디렉토리 기준으로 scripts/ 로 이동
cd "$SLURM_SUBMIT_DIR/scripts"

CHECKPOINT="/home/jinwoongjung/MTGS/experiments/vg_gaze_graph_v3(lah_min, monitor=social_ap)/train/checkpoints/best.ckpt"

EXP_NAME="reproduce_vg_gaze_graph_lahmin_socialap"

# NOTE: CHECKPOINT path contains spaces/parens/commas → must be single-quoted so
# Hydra treats it as a literal string value (not list/group syntax).
# model.weights=False skips the warm-start load (test.checkpoint overrides all weights anyway).
python -s ./main.py experiment.task=test \
    model.weights=False \
    gaze_graph.use=true \
    gaze_graph.laeo_derive=lah_min \
    "test.checkpoint='${CHECKPOINT}'" \
    "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"

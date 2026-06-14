#!/bin/bash

#SBATCH --job-name=vsgaze_train
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

CHECKPOINT="/home/jinwoongjung/MTGS/weights/mtgs-vsgaze.ckpt"

EXP_NAME="test_VSGaze_gaze_graph"

python -s ./main.py experiment.task=test \
    test.checkpoint=$CHECKPOINT \
    "hydra.run.dir=\${hydra:runtime.cwd}/../experiments/\${now:%Y-%m-%d}/${EXP_NAME}"

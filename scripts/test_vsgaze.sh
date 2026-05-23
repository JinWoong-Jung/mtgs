#!/bin/bash

#SBATCH --job-name=vsgaze_test
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -p gpu
#SBATCH --output=logs/vsgaze_test_%j.out
#SBATCH --error=logs/vsgaze_test_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR/scripts"

CHECKPOINT="/home/jinwoongjung/MTGS/experiments/2026-05-18/MTGS-dinov2-vitb14-448-VSGaze/train/checkpoints/best.ckpt"

python main.py experiment.task=test \
    test.checkpoint=$CHECKPOINT

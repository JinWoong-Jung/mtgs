#!/bin/bash
#SBATCH --job-name=vlm_extract
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=12:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=64G
#SBATCH --output=logs/vlm_extract_%j.out
#SBATCH --error=logs/vlm_extract_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

STAGE_A_CKPT="/path/to/stage_a_gaze_graph.ckpt"          # ← set before submitting
CACHE_DIR="/home/jinwoongjung/MTGS/data/vlm_feature_cache"

python extract_vlm_features.py \
    vlm.stage_a_ckpt="${STAGE_A_CKPT}" \
    vlm.feature_cache.dir="${CACHE_DIR}" \
    train.num_workers=4

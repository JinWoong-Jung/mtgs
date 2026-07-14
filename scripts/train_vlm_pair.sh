#!/bin/bash

#SBATCH --job-name=vlm_pair
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlmpair_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlmpair_%j.err

set -e

# Slurm normally separates stderr, but tqdm/Transformers/W&B use it for ordinary
# progress logs. Merge all ordinary stderr into .out; train_pair.py writes exactly one
# explicit metric block per completed epoch to this path instead.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export PAIR_EPOCH_METRICS_PATH="/home/jinwoongjung/MTGS/scripts/logs/vlmpair_${SLURM_JOB_ID}.err"
fi
exec 2>&1

CONFIG=${CONFIG:-mtgs/config/config_vlm_pair.yaml}
CACHE=${CACHE:-/home/jinwoongjung/MTGS/data/vlm_feature}
RESUME=${RESUME:-}

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p scripts/logs
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume "$RESUME")

python -m vlm.train_pair \
  --config "$CONFIG" \
  --manifest "$CACHE/manifest_train.jsonl" \
  --frame_root "$CACHE/overlays/train" \
  --graph_feats "$CACHE/vlmgraph_train.pt" \
  --val_manifest "$CACHE/manifest_val.jsonl" \
  --val_frame_root "$CACHE/overlays/val" \
  --val_graph_feats "$CACHE/vlmgraph_val.pt" \
  --val_gtmeta "$CACHE/gtmeta_val.pt" \
  "${RESUME_ARGS[@]}"

#!/bin/bash

#SBATCH --job-name=eval_vlm_pair
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/eval_vlmpair_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/eval_vlmpair_%j.err

set -e

MODE=${MODE:-raw_graph}
NAME=${NAME:-$MODE}
SPLIT=${SPLIT:-test}
CHECKPOINT=${CHECKPOINT:-}
CONFIG=${CONFIG:-mtgs/config/config_vlm_pair.yaml}
CACHE=${CACHE:-/home/jinwoongjung/MTGS/data/vlm_feature}
OUT=${OUT:-experiments/vlm_pair_eval/${NAME}_${SPLIT}}
DEVICE=${DEVICE:-auto}

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
CHECKPOINT_ARGS=()
if [ "$MODE" != "raw_graph" ]; then
    if [ -z "$CHECKPOINT" ]; then
        echo "CHECKPOINT is required for MODE=$MODE" >&2
        exit 2
    fi
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

python -m vlm.eval_pair run \
  --mode "$MODE" \
  --name "$NAME" \
  --config "$CONFIG" \
  --manifest "$CACHE/manifest_${SPLIT}.jsonl" \
  --frame_root "$CACHE/overlays/${SPLIT}" \
  --graph_feats "$CACHE/vlmgraph_${SPLIT}.pt" \
  --gtmeta "$CACHE/gtmeta_${SPLIT}.pt" \
  --output_dir "$OUT" \
  --device "$DEVICE" \
  "${CHECKPOINT_ARGS[@]}"

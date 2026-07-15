#!/bin/bash

#SBATCH --job-name=eval_vlm
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/eval_vlm_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/eval_vlm_%j.err

set -e

MODE=${MODE:-raw_graph}
NAME=${NAME:-$MODE}
SPLIT=${SPLIT:-test}
CHECKPOINT=${CHECKPOINT:-}
CONFIG=${CONFIG:-mtgs/config/config_vlm.yaml}
CACHE=${CACHE:-/home/jinwoongjung/MTGS/data/vlm_feature}
OUT=${OUT:-experiments/vlm_eval/${NAME}_${SPLIT}}
DEVICE=${DEVICE:-auto}
# Leave MANIFEST_PROFILE unset to read data.profile from CONFIG. MANIFEST remains
# an explicit escape hatch for evaluating an ad-hoc manifest.
MANIFEST_PROFILE=${MANIFEST_PROFILE:-}
MANIFEST=${MANIFEST:-}
FRAME_ROOT=${FRAME_ROOT:-$CACHE/overlays/${SPLIT}}

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
if [ -z "$MANIFEST_PROFILE" ]; then
    MANIFEST_PROFILE=$(python - "$CONFIG" <<'PYCFG'
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).get("data", {}).get("profile", "full"))
PYCFG
)
fi
MANIFEST=${MANIFEST:-$CACHE/manifests/$MANIFEST_PROFILE/manifest_${SPLIT}.jsonl}
if [ ! -f "$MANIFEST" ]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 2
fi
echo "[vlm] manifest_profile=$MANIFEST_PROFILE manifest=$MANIFEST"
CHECKPOINT_ARGS=()
if [ "$MODE" != "raw_graph" ]; then
    if [ -z "$CHECKPOINT" ]; then
        echo "CHECKPOINT is required for MODE=$MODE" >&2
        exit 2
    fi
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

python -m vlm.evaluate run \
  --mode "$MODE" \
  --name "$NAME" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --frame_root "$FRAME_ROOT" \
  --graph_feats "$CACHE/vlmgraph_${SPLIT}.pt" \
  --gtmeta "$CACHE/gtmeta_${SPLIT}.pt" \
  --output_dir "$OUT" \
  --device "$DEVICE" \
  "${CHECKPOINT_ARGS[@]}"

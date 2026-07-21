#!/bin/bash

#SBATCH --job-name=eval_vlm_bal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/eval_vlm_balanced_full_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/eval_vlm_balanced_full_%j.err

set -e

# Slurm keeps stdout/stderr separate: model loading and real errors land in .err,
# while our own tqdm bars and the final result table (printed via Python's stdout)
# land in .out.

# One-off sibling of eval_vlm.sh: eval_vlm.sh's RUN_DIR is a hardcoded literal
# and its test profile is always forced to "full" via input_provenance.env
# (VLM_TEST_MANIFEST_PROFILE=full, frozen by train_vlm.sh) -- there is no way
# to ask it for a balanced-profile test run. This script only exists to answer
# "what does VLM_v3(balanced) score on its own balanced test split" as a first
# quick check, ahead of the full-profile run.

# Set only the completed VLM training directory.
RUN_DIR="/home/jinwoongjung/MTGS/experiments/VLM/VLM_v7(balanced_full)-routing(0.8)"

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
MODE="vlm"
SPLIT="test"
NAME="$(basename "$RUN_DIR")_balanced_full"
CHECKPOINT="$RUN_DIR/train/checkpoints/best"
CONFIG="$RUN_DIR/config_vlm.yaml"
OUT="$RUN_DIR/test_balanced_full"
DEVICE="auto"
WANDB_OFF=1
PROVENANCE="$RUN_DIR/input_provenance.env"
if [ ! -f "$PROVENANCE" ]; then
    echo "Missing input provenance: $PROVENANCE" >&2
    echo "This historical run must be backfilled once with its original cache path." >&2
    exit 2
fi
source "$PROVENANCE"
if [ -z "$VLM_CACHE" ]; then
    echo "Invalid input provenance: $PROVENANCE" >&2
    exit 2
fi
CACHE="$VLM_CACHE"
# Deliberately override the provenance's always-"full" test profile with
# "balanced" -- that is this script's entire reason to exist.
MANIFEST_PROFILE="balanced_full"
MANIFEST="$CACHE/manifests/$MANIFEST_PROFILE/manifest_$SPLIT.jsonl"
FRAME_ROOT="$CACHE/overlays/$SPLIT"
for required in "$CONFIG" "$CHECKPOINT" "$MANIFEST" "$FRAME_ROOT" "$CACHE/vlmgraph_$SPLIT.pt" "$CACHE/gtmeta_$SPLIT.pt"; do
    if [ ! -e "$required" ]; then
        echo "Missing run input: $required" >&2
        exit 2
    fi
done
echo "[vlm] run=$RUN_DIR checkpoint=$CHECKPOINT cache=$CACHE test_profile=$MANIFEST_PROFILE output=$OUT"
echo "[vlm] manifest_profile=$MANIFEST_PROFILE manifest=$MANIFEST"

CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
WANDB_ARGS=()
[ "$WANDB_OFF" = "1" ] && WANDB_ARGS=(--wandb_off)

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
  "${CHECKPOINT_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

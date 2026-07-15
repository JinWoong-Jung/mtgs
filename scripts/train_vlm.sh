#!/bin/bash

#SBATCH --job-name=vlm
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlm_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlm_%j.err

set -e

# Slurm normally separates stderr, but tqdm/Transformers/W&B use it for ordinary
# progress logs. Merge all ordinary stderr into .out; train.py writes exactly one
# explicit metric block per completed epoch to this path instead.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export VLM_EPOCH_METRICS_PATH="/home/jinwoongjung/MTGS/scripts/logs/vlm_${SLURM_JOB_ID}.err"
fi
exec 2>&1

CONFIG=${CONFIG:-mtgs/config/config_vlm.yaml}
CACHE=${CACHE:-/home/jinwoongjung/MTGS/data/vlm_feature}
# A profile chooses the dataset subset. Leave MANIFEST_PROFILE unset to read
# data.profile from CONFIG; explicit environment overrides remain supported.
MANIFEST_PROFILE=${MANIFEST_PROFILE:-}
MANIFEST_ROOT=${MANIFEST_ROOT:-}
RESUME=${RESUME:-}
# Run the test set exactly once after successful training, using the validation-best
# checkpoint. Set TEST_EVAL=0 to train/validate only; these overrides are useful when
# a test run needs a smaller batch or a separate output directory.
TEST_EVAL=${TEST_EVAL:-1}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-0}
TEST_NUM_WORKERS=${TEST_NUM_WORKERS:--1}
TEST_OUT=${TEST_OUT:-}
TEST_NAME=${TEST_NAME:-}

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
MANIFEST_ROOT=${MANIFEST_ROOT:-$CACHE/manifests/$MANIFEST_PROFILE}
if [ ! -f "$MANIFEST_ROOT/manifest_train.jsonl" ] || [ ! -f "$MANIFEST_ROOT/manifest_val.jsonl" ] || [ ! -f "$MANIFEST_ROOT/manifest_test.jsonl" ]; then
    echo "Missing train/val/test manifests under MANIFEST_ROOT=$MANIFEST_ROOT" >&2
    exit 2
fi
echo "[vlm] manifest_profile=$MANIFEST_PROFILE manifest_root=$MANIFEST_ROOT"
RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume "$RESUME")

python -m vlm.train \
  --config "$CONFIG" \
  --manifest "$MANIFEST_ROOT/manifest_train.jsonl" \
  --frame_root "$CACHE/overlays/train" \
  --graph_feats "$CACHE/vlmgraph_train.pt" \
  --val_manifest "$MANIFEST_ROOT/manifest_val.jsonl" \
  --val_frame_root "$CACHE/overlays/val" \
  --val_graph_feats "$CACHE/vlmgraph_val.pt" \
  --val_gtmeta "$CACHE/gtmeta_val.pt" \
  "${RESUME_ARGS[@]}"

if [ "$TEST_EVAL" = "1" ]; then
    # train.py snapshots this config before training. Reuse that snapshot so the
    # best adapter is evaluated with exactly its original model/input settings.
    RUN_DIR=$(python - "$CONFIG" <<'PYCFG'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(f"{cfg.experiment.out_root}/{cfg.experiment.name}")
PYCFG
)
    RUN_CONFIG="$RUN_DIR/config_vlm.yaml"
    BEST_CHECKPOINT="$RUN_DIR/train/checkpoints/best"
    if [ ! -f "$RUN_CONFIG" ] || [ ! -d "$BEST_CHECKPOINT" ]; then
        echo "Missing completed run config or best checkpoint: run=$RUN_DIR" >&2
        exit 3
    fi
    if [ -z "$TEST_OUT" ]; then
        TEST_OUT="$RUN_DIR/test"
    fi
    if [ -z "$TEST_NAME" ]; then
        TEST_NAME="$(basename "$RUN_DIR")_${MANIFEST_PROFILE}"
    fi
    echo "[vlm] test: checkpoint=$BEST_CHECKPOINT manifest=$MANIFEST_ROOT/manifest_test.jsonl output=$TEST_OUT"
    python -m vlm.evaluate run \
      --mode vlm \
      --name "$TEST_NAME" \
      --checkpoint "$BEST_CHECKPOINT" \
      --config "$RUN_CONFIG" \
      --manifest "$MANIFEST_ROOT/manifest_test.jsonl" \
      --frame_root "$CACHE/overlays/test" \
      --graph_feats "$CACHE/vlmgraph_test.pt" \
      --gtmeta "$CACHE/gtmeta_test.pt" \
      --output_dir "$TEST_OUT" \
      --batch_size "$TEST_BATCH_SIZE" \
      --num_workers "$TEST_NUM_WORKERS" \
      --device auto
fi

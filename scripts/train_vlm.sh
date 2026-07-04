#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: JinWoong Jung <jinwoong1010@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vlm_stage2
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/vlm_%j.out
#SBATCH --error=logs/vlm_%j.err

# conda 환경 활성화 (user site-packages 무시하여 ~/.local 충돌 방지)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

# Navigate to MTGS repo ROOT so that both `vlm` and `mtgs` packages are importable.
# (train_vsgaze.sh cd's into scripts/; train_vlm.sh must run from the parent.)
if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi

# Fail fast: in multi-command modes (e.g. `eval` runs nograph then blend) a failed
# first step must abort, not silently feed the next step stale/missing artifacts.
set -e

# ── Configuration (override at submission time with: MODE=export sbatch ...) ──
MODE=${MODE:-graphtext}           # export | overlays | graphtext | token | eval
SPLIT=${SPLIT:-train}             # train | val | test
CHECKPOINT=${CHECKPOINT:-experiments/V14.5/train/checkpoints/best.ckpt}
EXPERIMENT=${EXPERIMENT:-B_graphtext}   # W&B run name + output sub-dir
BLIND=${BLIND:-0}                 # 1 → answer-blind ablation (variant D)
UPWEIGHT=${UPWEIGHT:-1}           # >1 → graph-wrong up-sampling (variant E)
CACHE=results/vlm_cache
OUT=experiments/vlm/$EXPERIMENT

mkdir -p "$CACHE" logs

# ── Pipeline stages ───────────────────────────────────────────────────────────
case $MODE in

  # Stage 1: export frozen-Stage-1 graph features to disk
  export)
    python -m vlm.graph_export \
      --split "$SPLIT" \
      --ckpt "$CHECKPOINT" \
      --out "$CACHE/vlmgraph_${SPLIT}.pt" \
      --batch_size 4 ;;

  # Stage 2a: render per-frame overlays + write manifest + gtmeta cache
  overlays)
    python -m vlm.data_prep overlays \
      --split "$SPLIT" \
      --out "$CACHE/overlays" \
      --manifest "$CACHE/manifest_${SPLIT}.jsonl" \
      --gtmeta "$CACHE/gtmeta_${SPLIT}.pt" ;;

  # Stage 2b-B/D/E: graph-text LoRA fine-tuning (nograph mode)
  #   B: defaults  (BLIND=0, UPWEIGHT=1)
  #   D: BLIND=1   (answer-blind)
  #   E: BLIND=1 UPWEIGHT=4  (graph-wrong up-sampled)
  graphtext)
    GRAPHTEXT_BLIND=$BLIND \
    GRAPH_WRONG_UPWEIGHT=$UPWEIGHT \
    python -m vlm.train nograph \
      --manifest "$CACHE/manifest_train.jsonl" \
      --overlay_dir "$CACHE/overlays/train" \
      --graph_feats "$CACHE/vlmgraph_train.pt" \
      --out "$OUT" \
      --val_manifest "$CACHE/manifest_val.jsonl" \
      --val_overlay_dir "$CACHE/overlays/val" \
      --val_gtmeta "$CACHE/gtmeta_val.pt" \
      --val_graph_feats "$CACHE/vlmgraph_val.pt" \
      --epochs 2 --bs 2 --accum 8 --lr 1e-4 \
      --wandb_name "$EXPERIMENT" ;;

  # Stage 2b-C: graph-token LoRA fine-tuning (token injection mode)
  token)
    python -m vlm.train token \
      --manifest "$CACHE/manifest_train.jsonl" \
      --overlay_dir "$CACHE/overlays/train" \
      --graph_feats "$CACHE/vlmgraph_train.pt" \
      --out "$OUT" \
      --val_manifest "$CACHE/manifest_val.jsonl" \
      --val_overlay_dir "$CACHE/overlays/val" \
      --val_gtmeta "$CACHE/gtmeta_val.pt" \
      --val_graph_feats "$CACHE/vlmgraph_val.pt" \
      --epochs 2 --bs 8 --lr 1e-4 \
      --wandb_name "$EXPERIMENT" ;;

  # Stage 3: evaluation + graph-VLM soft blend sweep
  eval)
    python -m vlm.eval nograph \
      --ckpt "$OUT/final" \
      --manifest "$CACHE/manifest_${SPLIT}.jsonl" \
      --overlay_dir "$CACHE/overlays/$SPLIT" \
      --gtmeta "$CACHE/gtmeta_${SPLIT}.pt" \
      --graph_feats "$CACHE/vlmgraph_${SPLIT}.pt" \
      --preds_out "$CACHE/preds_${EXPERIMENT}_${SPLIT}.pt"
    python -m vlm.eval blend \
      --feat "$CACHE/vlmgraph_${SPLIT}.pt" \
      --pvlm "$CACHE/preds_${EXPERIMENT}_${SPLIT}.pt" \
      --alphas 0,0.25,0.3,0.5,1.0 ;;

  *)
    echo "unknown MODE=$MODE (choices: export | overlays | graphtext | token | eval)"
    exit 1 ;;
esac

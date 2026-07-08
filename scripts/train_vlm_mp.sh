#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: JinWoong Jung <jinwoong1010@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vlm_mp
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlmmp_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlmmp_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# 실험 F: multi-person dense-head VLM 학습/평가 런처.
#   전제: graph 피처(vlmgraph_*.pt) + gtmeta + frame 이미지가 이미 CACHE 에 있어야 함
#         (오프라인 추출은 scripts/graph_extract.sh 담당).
#   MODE=train : 학습 → 끝나면 곧바로 test 평가까지 한 job 에서
#   MODE=eval  : 저장된 체크포인트로 SPLIT 평가만
#   하이퍼파라미터는 전부 CONFIG(yaml)에.
# ─────────────────────────────────────────────────────────────────────────────

# ── 설정 (여기만 바꾸면 됨) ──────────────────────────────────────────────────
MODE=train                              # train | eval
EXP_NAME="VLM_v1"                    # 저장위치: experiments/<날짜>/<EXP_NAME>/ (train_vsgaze.sh와 동일)
CONFIG=mtgs/config/config_vlm_mp.yaml
WHICH=best                              # 평가 대상 체크포인트: best | last
SPLIT=test                              # MODE=eval 일 때 평가할 split
CACHE=/home/jinwoongjung/MTGS/data/vlm_feature
# ─────────────────────────────────────────────────────────────────────────────

RUN_DIR="experiments/$(date +%Y-%m-%d)/${EXP_NAME}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 조각화 완화(OOM 방지)

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi
set -e
mkdir -p "$CACHE" /home/jinwoongjung/MTGS/scripts/logs

echo "===== 실험 F: $EXP_NAME -> $RUN_DIR ====="
case $MODE in
  train)
    python -u -m vlm.mp.train \
      --config "$CONFIG" \
      --run_dir "$RUN_DIR" \
      --wandb_name "$EXP_NAME" \
      --vlmgraph_train "$CACHE/vlmgraph_train.pt" \
      --gtmeta_train   "$CACHE/gtmeta_train.pt" \
      --overlay_train  "$CACHE/overlays/train" \
      --vlmgraph_val   "$CACHE/vlmgraph_val.pt" \
      --gtmeta_val     "$CACHE/gtmeta_val.pt" \
      --overlay_val    "$CACHE/overlays/val"
    echo "===== 학습 완료 -> test 평가 (WHICH=$WHICH) ====="
    python -u -m vlm.mp.eval --config "$CONFIG" --which "$WHICH" --run_dir "$RUN_DIR" \
      --vlmgraph "$CACHE/vlmgraph_test.pt" --gtmeta "$CACHE/gtmeta_test.pt" \
      --overlay_dir "$CACHE/overlays/test" \
      --preds_out "$CACHE/preds_mp_test.pt" || echo "[warn] test 평가 실패; 재실행: MODE=eval" ;;
  eval)
    python -u -m vlm.mp.eval --config "$CONFIG" --which "$WHICH" --run_dir "$RUN_DIR" \
      --split_tag "$SPLIT" \
      --vlmgraph "$CACHE/vlmgraph_${SPLIT}.pt" --gtmeta "$CACHE/gtmeta_${SPLIT}.pt" \
      --overlay_dir "$CACHE/overlays/$SPLIT" \
      --preds_out "$CACHE/preds_mp_${SPLIT}.pt" ;;
  *) echo "unknown MODE=$MODE (train | eval)"; exit 1 ;;
esac

#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: JinWoong Jung <jinwoong1010@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=vlm_frame
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlmframe_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlmframe_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# VLM Stage-2 FRAME pipeline 런처 (graph-residual pairwise head).
#   전제: graph_extract.sh 산출물이 CACHE 에 있어야 함 (token 파이프라인과 동일 캐시 재사용).
#   MODE=train : 학습(+RUN_TEST=true 면 BEST ckpt 로 test 자동평가) → blend 분석.
#   하이퍼파라미터·실험이름은 전부 CONFIG(yaml)에.
# ─────────────────────────────────────────────────────────────────────────────

# ── 설정 (여기만 바꾸면 됨) ──────────────────────────────────────────────────
MODE=train                                  # train (frame 파이프라인은 train 전용 런처)
CONFIG=mtgs/config/config_vlm_frame.yaml    # 실험이름 + 모든 하이퍼파라미터
RUN_TEST=true                               # 학습 후 test 자동평가 (frame: ~42k forward, 수십 분)
CACHE=/home/jinwoongjung/MTGS/data/vlm_feature   # graph_extract.sh 산출물 위치
# ─────────────────────────────────────────────────────────────────────────────

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi

set -e
mkdir -p "$CACHE" /home/jinwoongjung/MTGS/scripts/logs

NAME=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').experiment.name)")

# graph⊕VLM soft-blend α-스윕 (분석용; frame 파이프라인은 residual 구조라 참고용).
run_blend () {
  python -m vlm.eval blend \
    --feat "$CACHE/vlmgraph_$1.pt" \
    --pvlm "$CACHE/preds_${NAME}_$1.pt" \
    --alphas 0,0.25,0.3,0.5,1.0
}

TEST_ARGS=()
[ "$RUN_TEST" = "true" ] && TEST_ARGS=(
  --test_manifest    "$CACHE/manifest_test.jsonl"
  --test_overlay_dir "$CACHE/overlays/test"
  --test_gtmeta      "$CACHE/gtmeta_test.pt"
  --test_graph_feats "$CACHE/vlmgraph_test.pt"
  --test_preds_out   "$CACHE/preds_${NAME}_test.pt"
)

python -m vlm.train_frame \
  --config          "$CONFIG" \
  --manifest        "$CACHE/manifest_train.jsonl" \
  --overlay_dir     "$CACHE/overlays/train" \
  --graph_feats     "$CACHE/vlmgraph_train.pt" \
  --val_manifest    "$CACHE/manifest_val.jsonl" \
  --val_overlay_dir "$CACHE/overlays/val" \
  --val_gtmeta      "$CACHE/gtmeta_val.pt" \
  --val_graph_feats "$CACHE/vlmgraph_val.pt" \
  "${TEST_ARGS[@]}"

if [ "$RUN_TEST" = "true" ]; then
  echo "===== 학습+test 완료 → graph⊕VLM blend 분석 ====="
  run_blend test || echo "[warn] blend 실패 (test/* 는 W&B 에 이미 기록됨)"
else
  echo "===== 학습 완료 (test 생략) ====="
fi

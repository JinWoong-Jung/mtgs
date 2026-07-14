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
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlm_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlm_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# VLM Stage-2 학습/평가 런처.
#   전제: graph 피처 + frame 이미지가 CACHE 에 미리 추출돼 있어야 함
#         (오프라인 추출은 scripts/graph_extract.sh 담당).
#   실험이름·배치·lr·epoch 등 하이퍼파라미터는 전부 CONFIG(yaml)에.
#
#   MODE=train : 학습(+RUN_TEST=true 면 BEST ckpt 로 test 자동평가) → blend 분석.
#                test/* 는 train.py 가 같은 W&B run 에 직접 기록.
#   MODE=eval  : 학습 없이 저장된 체크포인트로 SPLIT 평가만 (token → blend).
# ─────────────────────────────────────────────────────────────────────────────

# ── 설정 (여기만 바꾸면 됨) ──────────────────────────────────────────────────
MODE=train                            # train | eval
CONFIG=mtgs/config/config_vlm.yaml    # 실험이름 + 모든 하이퍼파라미터
WHICH=best                            # 평가 대상 체크포인트: best | last
SPLIT=test                            # MODE=eval 일 때 평가할 split: train | val | test
RUN_TEST=true                         # MODE=train 종료 후 test 자동평가 (866k, 수 시간).
                                      #   하이퍼파라미터 탐색 런은 false 로 두고 val 만 확인.
CACHE=/home/jinwoongjung/MTGS/data/vlm_feature   # graph_extract.sh 산출물 위치
# ─────────────────────────────────────────────────────────────────────────────

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1              # ~/.local 사용자 패키지 무시
export XFORMERS_DISABLED=1

# repo ROOT 로 이동 (vlm/mtgs 패키지 import). scripts/ 에서 제출하면 부모로.
if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi

set -e   # 단계 하나라도 실패하면 즉시 중단
mkdir -p "$CACHE" /home/jinwoongjung/MTGS/scripts/logs

# 실험 이름(CONFIG 단일 소스) — preds 캐시 파일명 구성에만 사용.
NAME=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONFIG').experiment.name)")

# graph⊕VLM soft-blend α-스윕 (분석용). preds_${NAME}_<split>.pt 가 있어야 함.
run_blend () {
  python -m vlm.eval blend \
    --feat "$CACHE/vlmgraph_$1.pt" \
    --pvlm "$CACHE/preds_${NAME}_$1.pt" \
    --alphas 0,0.25,0.3,0.5,1.0
}

case $MODE in

  train)
    TEST_ARGS=()
    [ "$RUN_TEST" = "true" ] && TEST_ARGS=(
      --test_manifest    "$CACHE/manifest_test.jsonl"
      --test_overlay_dir "$CACHE/overlays/test"
      --test_gtmeta      "$CACHE/gtmeta_test.pt"
      --test_graph_feats "$CACHE/vlmgraph_test.pt"
      --test_preds_out   "$CACHE/preds_${NAME}_test.pt"
    )
    python -m vlm.train \
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
      echo "===== 학습 완료 (test 생략; 필요시 MODE=eval SPLIT=test 로 별도 실행) ====="
    fi ;;

  eval)
    python -m vlm.eval token \
      --config      "$CONFIG" --which "$WHICH" \
      --manifest    "$CACHE/manifest_${SPLIT}.jsonl" \
      --overlay_dir "$CACHE/overlays/$SPLIT" \
      --graph_feats "$CACHE/vlmgraph_${SPLIT}.pt" \
      --gtmeta      "$CACHE/gtmeta_${SPLIT}.pt" \
      --preds_out   "$CACHE/preds_${NAME}_${SPLIT}.pt"
    run_blend "$SPLIT" ;;

  *)
    echo "unknown MODE=$MODE (choices: train | eval)"; exit 1 ;;

esac

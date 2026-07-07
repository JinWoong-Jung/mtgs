#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: JinWoong Jung <jinwoong1010@gmail.com>
# SPDX-License-Identifier: GPL-3.0-only

#SBATCH --job-name=graph_extract
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/gextract_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/gextract_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# VLM Stage-2 오프라인 추출 런처 (graph 피처 + plain frame 이미지)
#   split 하나에 대해 두 단계를 돌린다:
#     export   → vlm.graph_export : frozen graph 피처 (v_src/v_tgt/edge…) [GPU]
#     overlays → vlm.data_prep    : 프레임당 plain frame.png + manifest + gtmeta [CPU]
#   아래 설정값을 편집하거나, 제출 시 환경변수로 덮어써서 쓴다:
#     SPLIT=val STAGE=export CHECKPOINT=weights/foo.ckpt sbatch graph_extract.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── 설정 (여기만 바꾸면 됨. 환경변수로도 덮어쓰기 가능) ────────────────────────
SPLIT=${SPLIT:-train}                              # train | val | test
STAGE=${STAGE:-both}                               # export | overlays | both
CHECKPOINT=${CHECKPOINT:-weights/mtgs-vsgaze.ckpt} # graph 피처 추출용 체크포인트 (repo root 기준)
NUM_PEOPLE=${NUM_PEOPLE:-all}                       # all(가변 N) | <정수>. all이면 export bs=1 강제
BATCH_SIZE=${BATCH_SIZE:-4}                         # export 배치 (NUM_PEOPLE=all 이면 코드가 1로 강제)
CACHE=${CACHE:-/home/jinwoongjung/MTGS/data/vlm_feature}   # 산출물 저장 루트
# ─────────────────────────────────────────────────────────────────────────────

# conda 환경 활성화 (user site-packages 무시하여 ~/.local 충돌 방지)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

# repo ROOT 로 이동 (vlm/mtgs 패키지 import 가능하도록). scripts/ 에서 제출하면 부모로.
if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi

# 한 단계라도 실패하면 즉시 중단 (export 실패 후 stale overlays 방지)
set -e

mkdir -p "$CACHE" /home/jinwoongjung/MTGS/scripts/logs

echo "===== graph_extract: SPLIT=$SPLIT STAGE=$STAGE CKPT=$CHECKPOINT NUM_PEOPLE=$NUM_PEOPLE bs=$BATCH_SIZE ====="
echo "===== out -> $CACHE ====="

run_export () {
  echo "----- [export] graph features: split=$SPLIT -----"
  python -u -m vlm.graph_export \
    --split "$SPLIT" \
    --ckpt "$CHECKPOINT" \
    --out "$CACHE/vlmgraph_${SPLIT}.pt" \
    --batch_size "$BATCH_SIZE" \
    --num_people "$NUM_PEOPLE"
  echo "----- [export] done -> $CACHE/vlmgraph_${SPLIT}.pt -----"
}

run_overlays () {
  echo "----- [overlays] plain frame.png + manifest + gtmeta: split=$SPLIT (CPU) -----"
  python -u -m vlm.data_prep overlays \
    --split "$SPLIT" \
    --out "$CACHE/overlays" \
    --manifest "$CACHE/manifest_${SPLIT}.jsonl" \
    --gtmeta "$CACHE/gtmeta_${SPLIT}.pt"
  echo "----- [overlays] done -> $CACHE/overlays/$SPLIT, manifest_${SPLIT}.jsonl, gtmeta_${SPLIT}.pt -----"
}

case $STAGE in
  export)   run_export ;;
  overlays) run_overlays ;;                # 주의: overlays 는 CPU 전용. GPU 슬롯을 놀리므로 export 없이 단독이면 CPU 파티션 권장.
  both)     run_export; run_overlays ;;
  *)        echo "unknown STAGE=$STAGE (choices: export | overlays | both)"; exit 1 ;;
esac

echo "===== graph_extract DONE (SPLIT=$SPLIT STAGE=$STAGE) ====="

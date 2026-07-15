#!/bin/bash
# Export "original MTGS" (gaze_graph.use=False: decoder_lah/decoder_sa,
# LAEO=min(LAH,LAH^T)) per-pair logits on the exact frames the VLM pipeline's
# balanced manifest references, so results are directly comparable to
# 2 (MTGS+graph), 3 (VLM-only) and 4 (VLM+graph) on the identical population.

#SBATCH --job-name=mtgs_origin_export
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -p gpu
#SBATCH --output=logs/mtgs_origin_export_%j.out
#SBATCH --error=logs/mtgs_origin_export_%j.err

SPLIT="${1:-test}"
INDICES_FILE="${2:-/home/jinwoongjung/MTGS/data/vlm_feature/manifests/balanced/frame_indices_${SPLIT}.txt}"
CKPT="${3:-/home/jinwoongjung/MTGS/weights/mtgs-vsgaze.ckpt}"
OUT="${4:-/home/jinwoongjung/MTGS/data/vlm_feature/mtgs_origin_${SPLIT}.pt}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

cd /home/jinwoongjung/MTGS

echo "[export_mtgs_origin] split=$SPLIT indices_file=$INDICES_FILE ckpt=$CKPT out=$OUT"

ARGS=(--split "$SPLIT" --ckpt "$CKPT" --out "$OUT")
if [ -f "$INDICES_FILE" ]; then
    ARGS+=(--indices_file "$INDICES_FILE")
else
    echo "[export_mtgs_origin] no indices file at $INDICES_FILE -> exporting the FULL split"
fi

python -m vlm.cache.baseline "${ARGS[@]}"

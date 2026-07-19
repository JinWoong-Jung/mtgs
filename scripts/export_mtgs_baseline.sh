#!/bin/bash
# Export logits from the original, graph-free MTGS baseline on VLM-manifest frames.

#SBATCH --job-name=mtgs_baseline_export
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -p gpu
#SBATCH --output=logs/mtgs_baseline_export_%j.out
#SBATCH --error=logs/mtgs_baseline_export_%j.err

SPLIT="${1:-test}"
INDICES_FILE="${2:-/home/jinwoongjung/MTGS/data/vlm_feature/manifests/balanced/frame_indices_${SPLIT}.txt}"
CKPT="${3:-/home/jinwoongjung/MTGS/weights/mtgs-vsgaze.ckpt}"
OUT="${4:-/home/jinwoongjung/MTGS/data/vlm_feature/mtgs_baseline_${SPLIT}.pt}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

cd /home/jinwoongjung/MTGS

echo "[export_mtgs_baseline] split=$SPLIT indices_file=$INDICES_FILE ckpt=$CKPT out=$OUT"

ARGS=(--split "$SPLIT" --ckpt "$CKPT" --out "$OUT")
if [ -f "$INDICES_FILE" ]; then
    ARGS+=(--indices_file "$INDICES_FILE")
else
    echo "[export_mtgs_baseline] no indices file at $INDICES_FILE -> exporting the full split"
fi

python -m vlm.cache.baseline "${ARGS[@]}"

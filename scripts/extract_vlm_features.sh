#!/bin/bash
#SBATCH --job-name=vlm_extract
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=12:00:00
#SBATCH -c 20
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/vlm_extract_%j.out
#SBATCH --error=logs/vlm_extract_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

STAGE_A_CKPT="/home/jinwoongjung/MTGS/experiments/vg_gaze_graph_v3(lah_min, monitor=social_ap)/train/checkpoints/best.ckpt"
BACKBONE="/home/jinwoongjung/QWEN_GazeEstimation/model/Qwen3-VL-4B-Instruct"
CACHE_DIR="/home/jinwoongjung/MTGS/data/vlm_feature_cache"
NUM_PEOPLE=4      # train/val keep up to 4 people; test uses "all"

# vlm.visual_encoder=true  → ALSO store the center-frame 448 image (float16) per
#                             clip (has_image). No vision tower here — it runs
#                             online at train time on the exact cached frame.
# vlm.visual_encoder=false → graph-only cache (no image; text+graph training).
VISUAL_ENCODER=false

# batch_size: MTGS processes B clips per GPU call. No vision tower at extraction,
# so per-batch time ≈ DataLoader (13-frame loads) + MTGS forward + h5 write.
BATCH_SIZE=32

# num_workers: each worker forks a copy of the annotation DataFrames (large).
# 16 workers × DataFrame size → RAM OOM. Keep low.
NUM_WORKERS=4

python extract_vlm_features.py \
    "vlm.stage_a_ckpt='${STAGE_A_CKPT}'" \
    "vlm.backbone='${BACKBONE}'" \
    vlm.feature_cache.dir="${CACHE_DIR}" \
    vlm.visual_encoder=${VISUAL_ENCODER} \
    +vlm.extract_batch_size=${BATCH_SIZE} \
    +vlm.skip_done=false \
    +train.num_workers=${NUM_WORKERS} \
    data.num_people=${NUM_PEOPLE}

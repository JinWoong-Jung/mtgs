#!/bin/bash
#SBATCH --job-name=vlm_align
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=72:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=logs/vlm_align_%j.out
#SBATCH --error=logs/vlm_align_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1
# reduce CUDA fragmentation (helps the per-QA-pair 8B forwards fit)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

STAGE_A_CKPT="/home/jinwoongjung/MTGS/experiments/vg_gaze_graph_v3(lah_min, monitor=social_ap)/train/checkpoints/best.ckpt"
CACHE_DIR="/home/jinwoongjung/MTGS/data/vlm_feature_cache_full_frame"
BACKBONE="/home/jinwoongjung/QWEN_GazeEstimation/model/Qwen3-VL-4B-Instruct"
EXP_NAME="vlm_v2"
EPOCHS=10
# visual_encoder=true → use the cached center-frame JPEG image (stored as
# compressed bytes in the h5 during extraction with visual_encoder=true).
# At train time the JPEG is decoded to PIL and fed directly to the Qwen3-VL
# vision tower (_encode_scene_pil), bypassing MTGS normalisation entirely.
# Requires the cache to have been extracted with visual_encoder=true (has_image_jpeg).
VISUAL_ENCODER="false"

# USE_ALL_FRAMES: true  → temporal mean of all T graph-feature frames is used
#                          as E_c / v_src_c / v_tgt_c in the graph tokenizer.
#                false → center frame only (t_c = T//2)  [default]
USE_ALL_FRAMES="false"

# ── Batch / memory ────────────────────────────────────────────────────────────
# train/val keep up to NUM_PEOPLE people (plus one padded/null slot internally),
# so cached train/val batches are uniform-N and BATCH_SIZE>1 works.
# cached test was extracted with num_people=all and is evaluated with batch_size=1.
# Memory still scales with (BATCH_SIZE × pairs-per-clip): the 8B is run once per
# QA pair and all pair graphs are kept until backward. If OOM, lower BATCH_SIZE.
BATCH_SIZE=8
ACCUM_GRAD=1          # effective batch = BATCH_SIZE × ACCUM_GRAD
NUM_PEOPLE=4          # train/val keep up to 4 people; test cache uses "all"

# ── Optimizer ─────────────────────────────────────────────────────────────────
LR=1e-5

# ── Scheduler (type: "cosine" | null) ────────────────────────────────────────
SCHED_TYPE="cosine"
WARMUP_EPOCHS=2
T_MAX_EPOCHS=${EPOCHS}   # cosine 주기; 보통 train epochs와 동일하게 설정

# Cached mode: first run scripts/extract_vlm_features.sh once. It writes
# train/val with NUM_PEOPLE and test with num_people=all, so vlm_align does not
# run MTGS online for train/val/test.
python main_vlm.py \
    experiment.name="${EXP_NAME}" \
    experiment.output_folder="../experiments/${EXP_NAME}" \
    "vlm.stage_a_ckpt='${STAGE_A_CKPT}'" \
    vlm.backbone="${BACKBONE}" \
    vlm.visual_encoder=${VISUAL_ENCODER} \
    +vlm.use_all_frames=${USE_ALL_FRAMES} \
    vlm.feature_cache.use=true \
    vlm.feature_cache.dir="${CACHE_DIR}" \
    vlm.optimizer.lr=${LR} \
    vlm.scheduler.type=${SCHED_TYPE} \
    vlm.scheduler.warmup_epochs=${WARMUP_EPOCHS} \
    vlm.scheduler.t_max_epochs=${T_MAX_EPOCHS} \
    train.epochs=${EPOCHS} \
    train.batch_size=${BATCH_SIZE} \
    val.batch_size=${BATCH_SIZE} \
    train.accumulate_grad_batches=${ACCUM_GRAD} \
    data.num_people=${NUM_PEOPLE}

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

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$SLURM_SUBMIT_DIR/scripts"
fi

STAGE_A_CKPT="/path/to/stage_a_gaze_graph.ckpt"  # ← set before submitting
EXP_NAME="vlm_align_v1"
EPOCHS=10

# Online mode (runs frozen MTGS every step). For the faster cached path, first
# run scripts/extract_vlm_features.py once, then add:
#     vlm.feature_cache.use=true vlm.feature_cache.dir=/path/to/cache
# (stage_a_ckpt is then ignored — the cache already encodes Stage A).
python main_vlm.py \
    experiment.name="${EXP_NAME}" \
    experiment.output_folder="../experiments/${EXP_NAME}" \
    vlm.stage_a_ckpt="${STAGE_A_CKPT}" \
    vlm.scheduler.t_max_epochs=${EPOCHS} \
    train.epochs=${EPOCHS} \
    train.batch_size=1 \
    train.accumulate_grad_batches=4 \
    train.num_workers=4 \
    data.num_people=11

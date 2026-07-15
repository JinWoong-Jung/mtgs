#!/bin/bash

# Rebuild the canonical VLM cache with a frozen MTGS-equivalent N=4 person
# selection for train/val while retaining the canonical N=all test cache.

#SBATCH --job-name=vlm_n4_cache
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=48:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlm_n4_cache_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlm_n4_cache_%j.err

set -euo pipefail

BASE_CACHE=${BASE_CACHE:-/home/jinwoongjung/MTGS/data/vlm_feature}
# Default to an in-place canonical rebuild. Set CACHE to another directory only
# when an isolated staging export is explicitly needed.
CACHE=${CACHE:-$BASE_CACHE}
CHECKPOINT=${CHECKPOINT:-/home/jinwoongjung/MTGS/experiments/V18/train/checkpoints/best.ckpt}
LAEO_DERIVE=${LAEO_DERIVE:-decoder}
SEED=${SEED:-101}
BATCH_SIZE=${BATCH_SIZE:-4}
# plans | metadata | graph | profiles | all
STAGE=${STAGE:-all}

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$(dirname "$SLURM_SUBMIT_DIR")"
else
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p "$CACHE" scripts/logs

link_existing() {
    local source="$1"
    local destination="$2"
    # In the canonical in-place mode, test/overlay source and destination are
    # already identical and need no link.
    if [ "$source" = "$destination" ]; then
        return
    fi
    if [ -L "$destination" ]; then
        if [ "$(readlink -f "$destination")" != "$(readlink -f "$source")" ]; then
            echo "Refusing to replace $destination: it points somewhere else" >&2
            exit 2
        fi
    elif [ -e "$destination" ]; then
        echo "Refusing to replace existing non-symlink $destination" >&2
        exit 2
    else
        ln -s "$source" "$destination"
    fi
}

run_plans() {
    for split in train val; do
        local plan="$CACHE/selection_${split}.json"
        if [ ! -f "$plan" ]; then
            python -m vlm.cache.selection \
                --split "$split" --num-people 4 --seed "$SEED" --out "$plan"
        fi
    done
}

prepare_shared_inputs() {
    link_existing "$BASE_CACHE/overlays" "$CACHE/overlays"
    for name in manifest_test.jsonl gtmeta_test.pt vlmgraph_test.pt; do
        link_existing "$BASE_CACHE/$name" "$CACHE/$name"
    done
}

run_metadata() {
    for split in train val; do
        python -m vlm.cache.render overlays \
            --split "$split" \
            --num_people 4 \
            --selection_plan "$CACHE/selection_${split}.json" \
            --out "$CACHE/overlays" \
            --manifest "$CACHE/manifest_${split}.jsonl" \
            --gtmeta "$CACHE/gtmeta_${split}.pt"
    done
}

run_graph() {
    for split in train val; do
        local pending="$CACHE/.vlmgraph_${split}.pt.pending"
        python -m vlm.cache.graph \
            --split "$split" \
            --ckpt "$CHECKPOINT" \
            --out "$pending" \
            --batch_size "$BATCH_SIZE" \
            --num_people 4 \
            --selection_plan "$CACHE/selection_${split}.json" \
            --laeo_derive "$LAEO_DERIVE"
        mv "$pending" "$CACHE/vlmgraph_${split}.pt"
    done
}

profile_split() {
    local profile="$1"
    local split="$2"
    shift 2
    python -m vlm.cache.manifest \
        --manifest "$CACHE/manifest_${split}.jsonl" \
        --gtmeta "$CACHE/gtmeta_${split}.pt" \
        --output "$CACHE/manifests/$profile/manifest_${split}.jsonl" \
        --report "$CACHE/manifests/$profile/report_${split}.json" \
        --seed "$SEED" "$@"
}

run_profiles() {
    mkdir -p "$CACHE/manifests/full"
    for split in train val test; do
        link_existing "$CACHE/manifest_${split}.jsonl" "$CACHE/manifests/full/manifest_${split}.jsonl"
    done
    for split in train val test; do
        profile_split fast_frame "$split" --frame-stride 3 --no-balance-labels
        profile_split balanced "$split" \
            --sources childplay videoattentiontarget --frame-stride 3
    done
}

case "$STAGE" in
    plans)
        run_plans ;;
    metadata)
        run_plans; prepare_shared_inputs; run_metadata ;;
    graph)
        run_plans; prepare_shared_inputs; run_graph ;;
    profiles)
        prepare_shared_inputs; run_profiles ;;
    all)
        run_plans; prepare_shared_inputs; run_metadata; run_graph; run_profiles ;;
    *)
        echo "Unknown STAGE=$STAGE (plans|metadata|graph|profiles|all)" >&2
        exit 2 ;;
esac

echo "[vlm-n4] complete: $CACHE (stage=$STAGE)"

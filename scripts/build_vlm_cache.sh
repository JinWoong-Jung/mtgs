#!/bin/bash

# Build a standalone VLM cache: train/val use the MTGS N<=4 policy; test uses all people. Use an empty CACHE for a clean rebuild.

#SBATCH --job-name=vlm_cache
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=96:00:00
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --mem=96G
#SBATCH --output=/home/jinwoongjung/MTGS/scripts/logs/vlm_cache_%j.out
#SBATCH --error=/home/jinwoongjung/MTGS/scripts/logs/vlm_cache_%j.err

set -euo pipefail

# set arguments
CACHE="/home/jinwoongjung/MTGS/data/graph_cache"  # output dir; point elsewhere for an isolated export, leave as-is for the canonical cache

CHECKPOINT="/home/jinwoongjung/MTGS/experiments/MTGS+graph/train/checkpoints/best.ckpt"  # MTGS+graph checkpoint to extract features from

LAEO_DERIVE="decoder"

SEED=101
BATCH_SIZE=8

# plans | metadata | boundary | graph | profiles | validate | all
STAGE="all"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1
export XFORMERS_DISABLED=1

# SLURM copies the submitted script into its own spool dir, so ${BASH_SOURCE[0]}
# does not point at this file under sbatch -- use $SLURM_SUBMIT_DIR (the actual
# directory sbatch was run from) instead, same pattern as train_vsgaze.sh.
if [ "$(basename "$SLURM_SUBMIT_DIR")" = "scripts" ]; then
    cd "$SLURM_SUBMIT_DIR/.."
else
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p "$CACHE" scripts/logs

link_canonical_manifest() {
    local source="$1"
    local destination="$2"
    if [ ! -f "$source" ]; then
        echo "Missing canonical manifest: $source" >&2
        exit 2
    fi
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


run_metadata() {
    for split in train val; do
        python -m vlm.cache.render overlays --split "$split" --num_people 4 --selection_plan "$CACHE/selection_$split.json" --out "$CACHE/overlays" --manifest "$CACHE/manifest_$split.jsonl" --gtmeta "$CACHE/gtmeta_$split.pt"
    done
    python -m vlm.cache.render overlays --split test --num_people all --out "$CACHE/overlays" --manifest "$CACHE/manifest_test.jsonl" --gtmeta "$CACHE/gtmeta_test.pt"
}

# Gaze-event boundary flags (Eyes on Gaze / EyeVLM-style transition-frame filtering).
# Graph-agnostic like run_metadata (CPU only, no checkpoint) -- consumed by the
# balanced/balanced_full profiles in run_profiles, not by full/fast_frame.
run_boundary() {
    for split in train val; do
        python -m vlm.cache.boundary --split "$split" --num_people 4 --selection_plan "$CACHE/selection_$split.json" --out "$CACHE/boundary_$split.pt"
    done
    python -m vlm.cache.boundary --split test --num_people all --out "$CACHE/boundary_test.pt"
}

run_graph() {
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Missing MTGS+graph checkpoint: $CHECKPOINT" >&2
        exit 2
    fi
    for split in train val; do
        local pending="$CACHE/.vlmgraph_$split.pt.pending"
        python -m vlm.cache.graph --split "$split" --ckpt "$CHECKPOINT" --out "$pending" --batch_size "$BATCH_SIZE" --num_people 4 --selection_plan "$CACHE/selection_$split.json" --laeo_derive "$LAEO_DERIVE"
        mv "$pending" "$CACHE/vlmgraph_$split.pt"
    done
    local pending="$CACHE/.vlmgraph_test.pt.pending"
    python -m vlm.cache.graph --split test --ckpt "$CHECKPOINT" --out "$pending" --batch_size "$BATCH_SIZE" --num_people all --laeo_derive "$LAEO_DERIVE"
    mv "$pending" "$CACHE/vlmgraph_test.pt"
}

profile_split() {
    local profile="$1"
    local split="$2"
    shift 2
    python -m vlm.cache.manifest --manifest "$CACHE/manifest_$split.jsonl" --gtmeta "$CACHE/gtmeta_$split.pt" --output "$CACHE/manifests/$profile/manifest_$split.jsonl" --report "$CACHE/manifests/$profile/report_$split.json" --seed "$SEED" "$@"
}

run_profiles() {
    mkdir -p "$CACHE/manifests/full"
    for split in train val test; do
        link_canonical_manifest "$CACHE/manifest_$split.jsonl" "$CACHE/manifests/full/manifest_$split.jsonl"
        profile_split fast_frame "$split" --frame-stride 3 --no-balance-labels
        profile_split balanced "$split" --sources childplay videoattentiontarget --frame-stride 3 --boundary-cache "$CACHE/boundary_$split.pt"
        profile_split balanced_full "$split" --frame-stride 3 --boundary-cache "$CACHE/boundary_$split.pt"
    done
}

validate_split() {
    local split="$1"
    if [ "$LAEO_DERIVE" = "decoder" ]; then
        python -m vlm.cache.validation --graph_feats "$CACHE/vlmgraph_$split.pt" --manifest "$CACHE/manifest_$split.jsonl" --gtmeta "$CACHE/gtmeta_$split.pt" --require_direct_laeo
    else
        python -m vlm.cache.validation --graph_feats "$CACHE/vlmgraph_$split.pt" --manifest "$CACHE/manifest_$split.jsonl" --gtmeta "$CACHE/gtmeta_$split.pt"
    fi
}

validate_cache() {
    for split in train val test; do
        if [ ! -d "$CACHE/overlays/$split" ]; then
            echo "Missing overlays: $CACHE/overlays/$split" >&2
            exit 2
        fi
        validate_split "$split"
    done
    for profile in full fast_frame balanced balanced_full; do
        for split in train val test; do
            if [ ! -f "$CACHE/manifests/$profile/manifest_$split.jsonl" ]; then
                echo "Missing profile manifest: $CACHE/manifests/$profile/manifest_$split.jsonl" >&2
                exit 2
            fi
        done
    done
}

case "$STAGE" in
    plans)
        run_plans ;;
    metadata)
        run_plans; run_metadata ;;
    boundary)
        run_plans; run_boundary ;;
    graph)
        run_plans; run_graph ;;
    profiles)
        run_profiles ;;
    validate)
        validate_cache ;;
    all)
        run_plans; run_metadata; run_boundary; run_graph; run_profiles; validate_cache ;;
    *)
        echo "Unknown STAGE=$STAGE (plans|metadata|boundary|graph|profiles|validate|all)" >&2
        exit 2 ;;
esac

echo "[vlm-cache] complete: $CACHE (stage=$STAGE)"

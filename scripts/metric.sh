#!/bin/bash
# Runs VSGaze test then computes post-processed metrics (Dist, AP_IO, F1_LAH, F1_LAEO, AP_SA).

#SBATCH --job-name=metric_calculation
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -p gpu
#SBATCH --output=logs/metric_calculation_%j.out

ARG="${1}"
if [ -z "$ARG" ]; then
    echo "Usage: bash metric.sh <checkpoint.ckpt | test_predictions.p>"
    exit 1
fi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mtgs
export PYTHONNOUSERSITE=1

if [[ "$ARG" == *.p ]]; then
    PRED="$ARG"
else
    MARKER=$(mktemp)
    cd /home/jinwoongjung/MTGS/scripts
    python main.py experiment.task=test test.checkpoint="$ARG"
    PRED=$(find /home/jinwoongjung/MTGS/experiments -name "test_predictions.p" -newer "$MARKER" 2>/dev/null | sort | tail -1)
    rm -f "$MARKER"
    if [ -z "$PRED" ]; then
        echo "ERROR: test_predictions.p not found."
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Metrics from: $PRED"
echo "========================================="

python - "$PRED" <<'EOF'
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, '/home/jinwoongjung/MTGS')
from mtgs.performance.compute_metrics import compute, CPU_Unpickler

results = []
with open(sys.argv[1], 'rb') as f:
    first = CPU_Unpickler(f).load()
    if isinstance(first, list):
        # legacy format: single pickle.dump of the full list
        results = first
    else:
        # streaming format: one pickle.dump per batch
        results.append(first)
        while True:
            try:
                results.append(CPU_Unpickler(f).load())
            except EOFError:
                break
compute(results, shuffle=False, thr=0.5)
EOF

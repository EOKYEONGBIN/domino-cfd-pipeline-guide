#!/bin/bash
# Runs a single DoMINO surface-prediction request on this machine (the A6000).
#
# Called remotely over SSH by the Kit-CAE "Request Prediction" extension
# (see kit_extension/domino_predict on the Windows laptop). Expects the
# Kit-CAE side to have already scp'd the input STL to:
#   ~/domino-ahmedml/requests/<request_id>/input/case/<name>.stl
# Writes the prediction to:
#   ~/domino-ahmedml/requests/<request_id>/output/prediction_0.vtp
# which the Kit-CAE side then scp's back and imports.
set -e

REQUEST_ID="$1"
# Faces / Streamlines are independently selectable from the Kit-CAE UI --
# skip the expensive full-mesh (surface, 4x subdivide) or full-grid (volume)
# computation for whichever one wasn't requested. Both default to true so a
# bare `run_prediction.sh <id>` (old callers, manual testing) is unchanged.
COMPUTE_SURFACE="${2:-true}"
COMPUTE_VOLUME="${3:-true}"
if [ -z "$REQUEST_ID" ]; then
    echo "Usage: run_prediction.sh <request_id> [compute_surface=true|false] [compute_volume=true|false]" >&2
    exit 1
fi

BASE=~/domino-ahmedml/requests/"$REQUEST_ID"
MODEL_DIR=~/domino-ahmedml/model
# AUDIT FIX (2026-10-01): there's only one GPU on this box. Without a lock,
# several trainees hitting "Request Prediction" around the same time would
# each spawn their own predict_on_stl.py, all loading the checkpoint onto
# the same A6000 at once -- risking CUDA OOM, and at best just slowing
# everyone down by fighting over the same SMs. flock below forces requests
# to queue and run one at a time instead.
GPU_LOCK_FILE=~/domino-ahmedml/requests/.gpu.lock

if [ ! -d "$BASE/input/case" ]; then
    echo "ERROR: $BASE/input/case not found (expected the input STL there)" >&2
    exit 1
fi
if ! ls "$MODEL_DIR"/*.mdlus >/dev/null 2>&1; then
    echo "ERROR: no .mdlus checkpoint found in $MODEL_DIR -- train a model first" >&2
    exit 1
fi

mkdir -p "$BASE/output"
source ~/venvs/domino/bin/activate
cd ~/physicsnemo/examples/cfd/external_aerodynamics/domino/src

flock "$GPU_LOCK_FILE" python predict_on_stl.py \
    --config-path ~/domino-ahmedml/configs \
    --config-name real_train_500 \
    eval.test_path="$BASE/input" \
    eval.save_path="$BASE/output" \
    resume_dir="$MODEL_DIR" \
    data.scaling_factors="$MODEL_DIR/scaling_factors.pkl" \
    eval.scaling_param_path="$MODEL_DIR" \
    hydra.run.dir="$BASE/hydra_run" \
    +eval.compute_surface="$COMPUTE_SURFACE" \
    +eval.compute_volume="$COMPUTE_VOLUME"

echo "=== PREDICTION DONE: $BASE/output ==="

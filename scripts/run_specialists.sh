#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_RUNTIME=${CAFETERIA_RUNTIME:-"$HOME/Library/Caches/yolo-cafeteria-runtime"}
TASK_STORE=${CAFETERIA_STORAGE:-"${TASK_ROOT}-assets"}
if [ -n "${CAFETERIA_PYTHON:-}" ]; then
    TASK_PYTHON=$CAFETERIA_PYTHON
elif [ -x "$TASK_ROOT/.venv/bin/python" ]; then
    TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
elif [ -x "$TASK_STORE/venv/bin/python" ]; then
    TASK_PYTHON="$TASK_STORE/venv/bin/python"
else
    TASK_PYTHON=$(command -v python3)
fi
export PYTHONPATH="$TASK_RUNTIME/py312:$TASK_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: run_specialists.sh {predict|evaluate|sort|explore|review44|precision|precision-classwise|hinge|replay|guard|train|preview} [options]

predict   Run the active four specialists.
evaluate  Score a fixed annotated cohort.
sort      Suggest disposal categories for individual detected objects.
explore   Run unchanged models on unverified review ZIP and render visual report.
review44  Score reviewed 44-image development diagnostics (V3 vs original).
precision Select a global-precision confidence threshold on validation; check fixed threshold on exposed test.
precision-classwise Optimize per-class confidence cutoffs on validation, preserving recall.
hinge     Train/evaluate a separate 3-class hinged-container candidate (never auto-promoted).
replay    Apply sorting rules to a frozen prediction receipt, without inference.
guard     Check a candidate evaluation for per-class validation regression.
train     Train a separate seven-class candidate with orientation augmentation.
preview   Inspect native training augmentation and transformed label boxes.

Use COMMAND --help for command-specific options. Training never auto-promotes models.
EOF
        exit 0 ;;
    sort)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/disposal.py" "$@" ;;
    explore)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/explore_candidates.py" "$@" ;;
    review44)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/evaluate_reviewed44.py" "$@" ;;
    precision)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/precision_sweep.py" "$@" ;;
    precision-classwise)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/precision_per_class.py" "$@" ;;
    hinge)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/hinged_candidate.py" "$@" ;;
    replay)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/replay_sorting.py" "$@" ;;
    guard)
        shift
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/guard_candidate.py" "$@" ;;
    train|preview)
        # Disable PyTorch's MPS allocation cap for the authorized training runs.
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/training.py" "$@" ;;
    *) exec "$TASK_PYTHON" "$TASK_ROOT/scripts/specialists.py" "$@" ;;
esac

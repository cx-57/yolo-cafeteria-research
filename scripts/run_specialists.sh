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
Usage: run_specialists.sh {predict|evaluate|train|preview} [options]

predict   Run the active four specialists.
evaluate  Score a fixed annotated cohort.
train     Train a separate seven-class candidate with orientation augmentation.
preview   Inspect native training augmentation and transformed label boxes.

Use COMMAND --help for command-specific options. Training never auto-promotes models.
EOF
        exit 0 ;;
    train|preview)
        # Disable PyTorch's MPS allocation cap for the authorized training runs.
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        exec "$TASK_PYTHON" "$TASK_ROOT/scripts/training.py" "$@" ;;
    *) exec "$TASK_PYTHON" "$TASK_ROOT/scripts/specialists.py" "$@" ;;
esac

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
exec "$TASK_PYTHON" "$TASK_ROOT/scripts/specialists.py" "$@"

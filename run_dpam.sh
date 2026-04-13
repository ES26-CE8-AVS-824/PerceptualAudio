#!/bin/bash
set -e

# =========================
# DPAM RUN CONFIG
# =========================

# Activate TF memory-safe GPU behavior (got TF crashes without it and GPU memory ¬100% usage)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Optional (recommended for cleaner logs)
export TF_CPP_MIN_LOG_LEVEL=2

# Ensure we are in project root
cd /workspace

# Activate virtual environment
source /opt/venv/bin/activate

echo "============================"
echo "Running DPAM test script..."
echo "Python: $(which python)"
echo "Venv: $VIRTUAL_ENV"
echo "GPU growth enabled: $TF_FORCE_GPU_ALLOW_GROWTH"
echo "============================"

# Run your script
python example_pip.py
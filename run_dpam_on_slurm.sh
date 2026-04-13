#!/bin/bash

#SBATCH --job-name=dpam
#SBATCH --output=logs/dpam_%j.out
#SBATCH --error=logs/dpam_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# #SBATCH --partition=<your_partition>   # Uncomment and set if required


PROJECT_ROOT="/ceph/project/es26-ce8-avs-824/whispers-in-the-storm"
SINGULARITY_CACHE="$HOME/.singularity"

DPAM_DIR="$PROJECT_ROOT/extern/PerceptualAudio"
DPAM_CONTAINER="$DPAM_DIR/dpam.sif"

CMD="$DPAM_DIR/run_dpam.sh"

(
cd "${DPAM_DIR}" || exit
singularity exec --nv \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${SINGULARITY_CACHE}:/scratch/singularity" \
    "${DPAM_CONTAINER}" \
    /bin/bash -c "
        set -euo pipefail && \
        export TMPDIR=/scratch/singularity/tmp && \
        export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
        export HF_DATASETS_CACHE=${DATA_ROOT}/hf_cache && \
        export TF_FORCE_GPU_ALLOW_GROWTH=true && \
        export TF_CPP_MIN_LOG_LEVEL=2 && \
        source /opt/venv/bin/activate && \
        ${CMD}
    "
)

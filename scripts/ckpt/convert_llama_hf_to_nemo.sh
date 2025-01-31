#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out
set -e

# open file limit
ulimit -n 65536 1048576

BASE_DIR=$(pwd)

source .venv/bin/activate

export TMPDIR="${BASE_DIR}/tmp"

INPUT_HF_NAME_OR_PATH=$1
OUPUT_NEMO_PATH=$2
HPARAMS_FILE=$3

export NVIDIA_PYTORCH_VERSION=""

# run
python scripts/ckpt/convert_llama_hf_to_nemo.py \
  --input-name-or-path ${INPUT_HF_NAME_OR_PATH} \
  --output-path ${OUPUT_NEMO_PATH} \
  --hparams-file ${HPARAMS_FILE} \
  --cpu-only \
  --n-jobs 96

echo "Extracting the Nemo checkpoint to ${OUPUT_NEMO_PATH}"
mkdir -p "${OUPUT_NEMO_PATH}"
tar -xvf "${OUPUT_NEMO_PATH}.nemo" -C "${OUPUT_NEMO_PATH}"

if [ -f "${OUPUT_NEMO_PATH}/model_config.yaml" ] && [ -d "${OUPUT_NEMO_PATH}/model_weights" ]; then
  echo "Successfully converted the checkpoint to Nemo format. Removing the nemo file."
  rm "${OUPUT_NEMO_PATH}.nemo"
fi

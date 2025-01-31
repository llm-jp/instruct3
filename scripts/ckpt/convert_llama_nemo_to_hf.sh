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

export NVIDIA_PYTORCH_VERSION=""

INPUT_NEMO_PATH=$1
OUTPUT_HF_PATH=$2
CHAT_TEMPLATE="{{bos_token}}{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"

MODEL_ID=$(basename ${INPUT_NEMO_PATH}) # The lowest directory name in INPUT_NEMO_PATH is used as the model id.

if [ -d "${INPUT_NEMO_PATH}/checkpoints" ]; then
  INPUT_NEMO_PATH="${INPUT_NEMO_PATH}/checkpoints"
fi

if [ ! -f "${INPUT_NEMO_PATH}/model_config.yaml" ]; then
  echo "model_config.yaml not found in ${INPUT_NEMO_PATH}"
  exit 1
fi

if [ ! -d "${INPUT_NEMO_PATH}/model_weights" ]; then
  ln -s $(ls -d ${INPUT_NEMO_PATH}/step=*-last) ${INPUT_NEMO_PATH}/model_weights
fi


# run
python scripts/ckpt/convert_llama_nemo_to_hf.py \
  --input-name-or-path ${INPUT_NEMO_PATH} \
  --chat-template "${CHAT_TEMPLATE}" \
  --output-hf-path ${OUTPUT_HF_PATH} \
  --cpu-only \
  --n-jobs 96

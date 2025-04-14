#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-small
#SBATCH --exclusive
#SBATCH --nodes 8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

set -e

# open file limit
ulimit -n 65536 1048576

source .venv/bin/activate

BASE_DIR=$(pwd)
export TMPDIR="${BASE_DIR}/tmp"

# distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# GPU settings
export NUM_GPU_PER_NODE=8
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((NUM_NODES * NUM_GPU_PER_NODE))
echo "NUM_NODES=${NUM_NODES}, NUM_GPU_PER_NODE=${NUM_GPU_PER_NODE}, NUM_GPUS=${NUM_GPUS}"

# set NVIDIA_PYTORCH_VERSION
export NVIDIA_PYTORCH_VERSION=""

NAME="sft-"$(tr -dc 0-9A-Za-z < /dev/urandom | fold -w 10 | head -1)
MODEL=v3-172b
MODEL_PATH=$1 # path to the model (nemo format)

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -bind-to none -map-by slot \
  -x PATH \
  python train_sft.py \
  trainer.num_nodes=${NUM_NODES} \
  use_mpi=True \
  use_slurm=True \
  name=${NAME} \
  trainer.sft.max_epochs=1 \
  mbs=1 \
  lr=0.00001 \
  min_lr=0.000001 \
  model=${MODEL} \
  model.restore_from_path=${MODEL_PATH}

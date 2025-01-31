#!/bin/bash

# python
python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel cython
pip install setuptools==69.5.1
pip install packaging
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cu118.txt
pip install git+https://github.com/NVIDIA/NeMo.git@0504c927afe61ddd40ade18f8553d0f52a65f509

# apex install
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout b7a4acc1
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
cd ../

# transformer engine install
PATH=/usr/local/cuda/bin:$PATH pip install git+https://github.com/NVIDIA/TransformerEngine.git@c81733f1032a56a817b594c8971a738108ded7d0 --no-cache-dir

# flash-attention install
git clone https://github.com/Dao-AILab/flash-attention -b v2.4.2
cd flash-attention
pip install -e .
cd ../

# make log and tmp directory
mkdir -p outputs
mkdir -p tmp

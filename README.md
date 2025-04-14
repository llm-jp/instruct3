# Instruct3

This repository contains the scripts for the supervised fine-tuning and direct preference optimization of the LLM-jp model.

## Preparation

### Setup Environment

See `install_cu118.sh` for CUDA 11.8 and `install_cu12.1.sh` for CUDA 12.1.

### Prepare Data

Run the following command to download and preprocess the data (about 4GB in total).

```bash
python preprocess_sft.py --dataset-dir /path/to/dataset

python preprocess_dpo.py --dataset-dir /path/to/dataset
```

### Prepare Config File

Please copy base_template.yaml to base.yaml.

```bash
cp configs/base_template.yaml configs/base.yaml
```
After copying, you need to modify the values `work_dir` and `data_dir` in `configs/base.yaml`.
`work_dir` is the directory where the model and the log files are stored and `data_dir` is the directory where the input data files are stored. This is the same as the `--dataset-dir` option in the preprocessing script.

## Checkpoint Conversion

### Hugging Face -> Nemo

`scripts/ckpt/convert_llama_hf_to_nemo.sh` converts the Hugging Face checkpoint to the Nemo checkpoint.
You may need to change ``job-name`` before running the script.

```bash
sbatch scripts/ckpt/convert_llama_hf_to_nemo.sh ${INPUT_HF_NAME_OR_PATH} ${OUTPUT_NEMO_PATH} ${HPARAMS_FILE}
```

#### Sample Code
```bash
# convert llm-jp-3-1.8b
sbatch scripts/ckpt/convert_llama_hf_to_nemo.sh llm-jp/llm-jp-3-1.8b /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-1.8b ./megatron_configs/llm-jp-3-1.8b.yaml
# convert llm-jp-3-3.7b
sbatch scripts/ckpt/convert_llama_hf_to_nemo.sh llm-jp/llm-jp-3-3.7b /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-3.7b ./megatron_configs/llm-jp-3-3.7b.yaml
# convert llm-jp-3-7.2b
sbatch scripts/ckpt/convert_llama_hf_to_nemo.sh llm-jp/llm-jp-3-7.2b /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-7.2b ./megatron_configs/llm-jp-3-7.2b.yaml
# convert llm-jp-3-13b
sbatch scripts/ckpt/convert_llama_hf_to_nemo.sh llm-jp/llm-jp-3-13b /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-13b ./megatron_configs/llm-jp-3-13b.yaml
# convert llm-jp-3-172b
sbatch scripts/ckpt/convert_llama_hf_to_nemo.sh llm-jp/llm-jp-3-172b /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-172b ./megatron_configs/llm-jp-3-172b.yaml
```

### Nemo -> Hugging Face

`scripts/ckpt/convert_llama_nemo_to_hf.sh` converts the Nemo checkpoint to the Hugging Face checkpoint.
You may need to change ``job-name`` before running the script.

**Note**: Use the absolute path for the input and output paths.

```bash
sbatch scripts/ckpt/convert_llama_nemo_to_hf.sh ${INPUT_NEMO_PATH} ${OUTPUT_HF_PATH}
```

#### Sample Code
```bash
# convert `sft-model1` model
sbatch scripts/ckpt/convert_llama_nemo_to_hf.sh /path/to/checkpoints/hf-to-nemo/sft-model1 /path/to/checkpoints/nemo-to-hf/sft-model1
```

## Supervised Fine-tuning

You may need to change ``job-name`` before running the script.

```bash
sbatch scripts/train/sft_1.8b.sh ${INPUT_NEMO_PATH}
```

### Sample Code
```bash
# 1.8B model with 2 nodes (16 GPUs)
sbatch --nodes 2 scripts/train/sft_1.8b.sh /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-1.8b
# 7.2B model with 4 nodes (32 GPUs)
sbatch --nodes 4 scripts/train/sft_7.2b.sh /path/to/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-7.2b
```

## Direct Preference Optimization

You may need to change ``job-name`` before running the script.

```bash
sbatch scripts/train/dpo_1.8b.sh ${INPUT_NEMO_PATH}
```

### Sample Code
```bash
# 1.8B model with 2 nodes (16 GPUs)
sbatch --nodes 2 scripts/train/dpo_1.8b.sh /path/to/checkpoints/hf-to-nemo/sft-model1
# 7.2B model with 4 nodes (32 GPUs)
sbatch --nodes 4 scripts/train/dpo_7.2b.sh /path/to/checkpoints/hf-to-nemo/sft-model2
```

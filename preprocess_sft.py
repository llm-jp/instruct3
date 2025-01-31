import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

import pandas as pd
from datasets import Dataset, load_dataset

NO_INPUT_PROMPT: str = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"


def save_sample(
    saved_samples: list[dict[str, Union[str, list[dict[str, str]]]]],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for sample in saved_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {int(len(saved_samples))} samples to {output_path}")


def process_default(
    dataset: Dataset,
    dataset_dir: Path,
    dataset_name: str,
    message_key: str = "messages",
):
    saved_samples: list[dict[str, Union[str, list[dict[str, str]]]]] = []
    sample_idx: int = 0
    for sample in dataset:
        messages = [{"role": "system", "content": NO_INPUT_PROMPT}]
        for message in sample[message_key]:
            messages.append(message)
        saved_samples.append(
            {
                "ID": f"{dataset_name}-{sample_idx}",
                "messages": messages,
            }
        )

    random.seed(42)
    random.shuffle(saved_samples)
    output_path: Path = dataset_dir / "train" / f"{dataset_name}.jsonl"
    save_sample(saved_samples, output_path)


def process_auto_multi_turn_by_calm3(dataset_dir: Path):
    raw_filepath: Path = dataset_dir / "AutoMultiTurnByCalm3-22B.jsonl"
    if not raw_filepath.exists():
        urlretrieve(
            "https://huggingface.co/datasets/kanhatakeyama/AutoMultiTurnByCalm3-22B/resolve/main/data/split_20240717_185452_0.jsonl",
            str(raw_filepath),
        )

    with raw_filepath.open(encoding="utf-8") as f:
        loaded_samples: list[dict] = [json.loads(line) for line in f]

    sample_idx: int = 0
    saved_samples: list[dict[str, Union[str, list[dict[str, str]]]]] = []
    for loaded_sample in loaded_samples:
        messages = [
            {"role": "system", "content": NO_INPUT_PROMPT},
            {"role": "user", "content": loaded_sample["q1"]},
            {"role": "assistant", "content": loaded_sample["a1"]},
        ]
        saved_samples.append(
            {
                "ID": f"multiturn_calm3-{sample_idx}",
                "messages": messages,
            }
        )
        sample_idx += 1

    random.seed(42)
    random.shuffle(saved_samples)
    output_path: Path = dataset_dir / "train" / "multiturn_calm3.jsonl"
    save_sample(saved_samples, output_path)


def process_random_to_fixed_multiturn_calm3(dataset_dir: Path):
    raw_filepath: Path = dataset_dir / "ramdom-to-fixed-multiturn-Calm3.parquet"
    if not raw_filepath.exists():
        urlretrieve(
            "https://huggingface.co/datasets/kanhatakeyama/ramdom-to-fixed-multiturn-Calm3/resolve/main/data/20240806filtered-00000-of-00001.parquet",
            str(raw_filepath),
        )

    df = pd.read_parquet(str(raw_filepath), engine="pyarrow")
    saved_samples: list[dict[str, Union[str, list[dict[str, str]]]]] = []
    for index, sample in df.iterrows():
        messages = [{"role": "system", "content": NO_INPUT_PROMPT}]
        messages.extend(sample["messages"])
        saved_samples.append(
            {
                "ID": f"random_to_fixed_multiturn_calm3-{index}",
                "messages": messages,
            }
        )

    random.seed(42)
    random.shuffle(saved_samples)
    output_path: Path = dataset_dir / "train" / "random_to_fixed_multiturn_calm3.jsonl"
    save_sample(saved_samples, output_path)


def process_daring_anteater(dataset_dir: Path):
    raw_filepath: Path = dataset_dir / "Daring-Anteater.jsonl"
    if not raw_filepath.exists():
        urlretrieve(
            "https://huggingface.co/datasets/nvidia/Daring-Anteater/resolve/main/train.jsonl",
            str(raw_filepath),
        )

    with raw_filepath.open(encoding="utf-8") as f:
        loaded_samples: list[dict] = [json.loads(line) for line in f]

    sample_idx: int = 0
    saved_samples: list[dict[str, Union[str, list[dict[str, str]]]]] = []
    for loaded_sample in loaded_samples:
        system_message: str = (
            loaded_sample["system"] if loaded_sample["system"] else NO_INPUT_PROMPT
        )
        messages = [{"role": "system", "content": system_message}]
        assert loaded_sample["mask"] == "User"
        for utterance in loaded_sample["conversations"]:
            if utterance["from"] == "User":
                messages.append({"role": "user", "content": utterance["value"]})
            elif utterance["from"] == "Assistant":
                messages.append({"role": "assistant", "content": utterance["value"]})
            else:
                raise ValueError(f"Invalid role: {utterance['from']}")
        saved_samples.append(
            {
                "ID": f"daring_anteater_en-{sample_idx}",
                "messages": messages,
            }
        )
        sample_idx += 1

    random.seed(42)
    random.shuffle(saved_samples)
    output_path: Path = dataset_dir / "train" / "daring_anteater_en.jsonl"
    save_sample(saved_samples, output_path)


def process_answer_carefully(dataset_dir: Path):
    dataset = load_dataset(
        "llm-jp/AnswerCarefully", data_dir="v2.0", split="validation"
    )

    saved_samples: list[dict[str, Union[str, list[dict[str, str]]]]] = []
    for sample in dataset:
        saved_samples.append(
            {
                "ID": sample["ID"],
                "messages": [
                    {"role": "system", "content": NO_INPUT_PROMPT},
                    {"role": "user", "content": sample["text"]},
                    {"role": "assistant", "content": sample["output"]},
                ],
            }
        )

    random.seed(42)
    random.shuffle(saved_samples)
    output_path: Path = dataset_dir / "train" / "ac_002.jsonl"
    save_sample(saved_samples, output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="./instruct3_datasets")
    args = parser.parse_args()

    dataset_dir: Path = Path(args.dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)

    # kanhatakeyama/AutoMultiTurnByCalm3-22B
    process_auto_multi_turn_by_calm3(dataset_dir)

    # kanhatakeyama/ramdom-to-fixed-multiturn-Calm3
    process_random_to_fixed_multiturn_calm3(dataset_dir)

    # llm-jp/wizardlm8x22b-logical-math-coding-sft-ja
    process_default(
        load_dataset("llm-jp/wizardlm8x22b-logical-math-coding-sft-ja", split="train"),
        dataset_dir,
        "logical_math_coding_wizard8x22b",
    )

    # llm-jp/magpie-sft-v1.0
    process_default(
        load_dataset(
            "llm-jp/magpie-sft-v1.0", split="train", revision="refs/convert/parquet"
        ),
        dataset_dir,
        "magpie_sft_v1.0",
        message_key="conversations",
    )

    # nvidia/Daring-Anteater
    process_daring_anteater(dataset_dir)

    # llm-jp/FLAN
    process_default(
        load_dataset("llm-jp/FLAN", split="train"),
        dataset_dir,
        "flan",
    )

    # llm-jp/Synthetic-JP-EN-Coding-Dataset
    process_default(
        load_dataset("llm-jp/Synthetic-JP-EN-Coding-Dataset", split="train"),
        dataset_dir,
        "synthetic_jp_en_coding",
    )

    # llm-jp/AnswerCarefully
    process_answer_carefully(dataset_dir)


if __name__ == "__main__":
    main()

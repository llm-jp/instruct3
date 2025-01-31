import json
import random
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset


def make_dpo_samples(dataset):
    dpo_samples = []
    for sample in dataset["train"]:
        dpo_samples.append(
            {
                "prompt": "<s>以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
                + sample["prompt"]
                + "\n\n### 応答:\n",
                "chosen_response": sample["chosen"],
                "rejected_response": sample["rejected"],
            }
        )
    random.seed(42)
    random.shuffle(dpo_samples)

    return dpo_samples


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="./instruct3_datasets")
    args = parser.parse_args()

    # ac-self-inst
    ac_self_inst = load_dataset("llm-jp/ac-self-inst")
    ac_self_inst_samples: list[dict] = make_dpo_samples(ac_self_inst)

    # aya-ja-evol-inst
    aya_ja_evol_inst_orig = load_dataset(
        "weblab-GENIAC/aya-ja-evol-instruct-calm3-dpo-masked"
    )
    idx2prompt: dict[int] = {}
    for sample in aya_ja_evol_inst_orig["train"]:
        idx2prompt[sample["idx"]] = sample["prompt"][1]["content"]

    aya_ja_evol_inst = load_dataset("llm-jp/aya-ja-evol-inst")
    aya_ja_evol_inst_samples = []
    for sample in aya_ja_evol_inst["train"]:
        prompt: str = idx2prompt[sample["idx"]]
        aya_ja_evol_inst_samples.append(
            {
                "prompt": "<s>以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
                + prompt
                + "\n\n### 応答:\n",
                "chosen_response": sample["chosen"],
                "rejected_response": sample["rejected"],
            }
        )
    random.seed(42)
    random.shuffle(aya_ja_evol_inst_samples)

    dev_samples = (
        ac_self_inst_samples[: int(len(ac_self_inst_samples) * 0.05)]
        + aya_ja_evol_inst_samples[: int(len(aya_ja_evol_inst_samples) * 0.05)]
    )
    print(
        f"Saving {len(dev_samples)} samples to {args.dataset_dir}/preference/dev/dev.jsonl"
    )
    dev_dir: Path = Path(f"{args.dataset_dir}/preference/dev")
    dev_dir.mkdir(exist_ok=True, parents=True)
    with (dev_dir / "dev.jsonl").open("w", encoding="utf-8") as f:
        for sample in dev_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    train_samples = (
        ac_self_inst_samples[int(len(ac_self_inst_samples) * 0.05) :]
        + aya_ja_evol_inst_samples[int(len(aya_ja_evol_inst_samples) * 0.05) :]
    )
    print(
        f"Saving {len(train_samples)} samples to {args.dataset_dir}/preference/train/train.jsonl"
    )
    train_dir: Path = Path(f"{args.dataset_dir}/preference/train")
    train_dir.mkdir(exist_ok=True, parents=True)
    with (train_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for sample in train_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()

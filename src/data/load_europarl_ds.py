#!/usr/bin/env python
"""This script download europarl-ner dataset from HF hub and
preprocess it in order to use it as a dataset for evaluation of
different projection methods.
"""

import argparse

import datasets

ID_TO_LABEL = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def tags_to_entity(row):
    tags = row["ner_tags"]
    entities = []
    start_idx = -1
    last_label = "O"
    for idx, tag in enumerate(tags):
        label = ID_TO_LABEL[tag]
        if label == "O" and start_idx != -1:
            entities.append(
                {
                    "start_idx": start_idx,
                    "end_idx": idx,
                    "label": last_label,
                }
            )
            start_idx = -1
        elif label.startswith("B-"):
            last_label = label[2:]
            start_idx = idx

    if start_idx != -1:
        entities.append(
            {
                "start_idx": start_idx,
                "end_idx": len(tags),
                "label": last_label,
            }
        )

    return {"src_entities": entities}


def load_dataset(args: argparse.Namespace) -> None:
    en_ds = datasets.load_dataset("ShkalikovOleh/europarl-ner", "en")["test"]
    en_ds = en_ds.map(tags_to_entity, remove_columns="ner_tags")
    en_ds = en_ds.rename_column("tokens", "src_words")
    en_ds.save_to_disk(args.out_path + "/en")

    if args.load_tgt_langs:
        for lang in ["de", "it", "es"]:
            lang_ds = datasets.load_dataset("ShkalikovOleh/europarl-ner", lang)["test"]
            lang_ds.save_to_disk(args.out_path + f"/{lang}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess europarl-ner dataset"
    )
    parser.add_argument("--load_tgt_langs", action=argparse.BooleanOptionalAction)
    parser.add_argument("out_path", type=str, help="path to the output directory")
    args = parser.parse_args()

    load_dataset(args)

import argparse
from itertools import groupby
from datasets import load_dataset, load_from_disk


def main(args: argparse.Namespace) -> None:
    if args.local_ds:
        ds = load_from_disk(args.dataset_name_or_path)
        if args.dataset_split is not None:
            ds = ds[args.dataset_split]
    else:
        ds = load_dataset(
            args.dataset_name_or_path, args.dataset_config, split=args.dataset_split
        )

    tags2labels = ds.features[args.entities_key].feature.names

    def iob_to_simple(label: str):
        if label.startswith("B-"):
            return label[2:]
        elif label.startswith("I-"):
            return label[2:]
        else:
            return label

    def count_entities(entities: list[int]) -> None:
        max_words = 0

        def key_func(tag: int) -> str:
            label = tags2labels[tag]
            return iob_to_simple(label)

        for key, group in groupby(entities, key_func):
            if key != "O":
                N = len(list(group))
                if N > max_words:
                    max_words = N

        return {"max_entity_length": max_words}

    ds = ds.map(count_entities, input_columns=[args.entities_key])
    print(
        ds.to_pandas()["max_entity_length"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
        )
    )

    # print("Maximum lenght of entity in words equals to:",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count a maximum lenght of entities (in words) in the dataset"
    )

    parser.add_argument("--dataset-name-or-path", "-d", type=str, required=True)
    parser.add_argument("--dataset-config", "-c", type=str)
    parser.add_argument("--dataset-split", "-s", type=str)
    parser.add_argument(
        "--local-ds", type=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--entities-key", type=str, default="ner_tags")

    args = parser.parse_args()
    main(args)

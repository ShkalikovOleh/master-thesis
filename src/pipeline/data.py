"""This module contains pipeline steps for loading and writing
HF datasets"""

import os

import datasets


class LoadDataset:
    def __init__(
        self,
        dataset_path: str,
        cfg_name: str | None = None,
        split: str | None = None,
        streaming: bool = False,
        local: bool = False,
    ) -> None:
        self.ds_path = dataset_path
        self.cfg_name = cfg_name
        self.split = split
        self.streaming = streaming
        self.local = local

    def __call__(self) -> datasets.Dataset:
        if self.split == "MERGE_ALL":
            asked_split = None
        else:
            asked_split = self.split

        if not self.local:
            ds = datasets.load_dataset(
                self.ds_path,
                name=self.cfg_name,
                split=asked_split,
                streaming=self.streaming,
            )
        else:
            if os.path.isfile(self.ds_path):
                ds = datasets.Dataset.from_file(self.ds_path)
            else:
                ds = datasets.load_from_disk(self.ds_path)
            if asked_split is not None:
                ds = ds[asked_split]

        if self.split == "MERGE_ALL" and isinstance(ds, datasets.DatasetDict):
            ds = datasets.concatenate_datasets([ds[key] for key in ds.keys()])

        return ds


class WriteDataset:
    def __init__(self, save_path: str, columns: list[str] | None = None) -> None:
        self.path = save_path
        self.columns = columns

    def __call__(self, ds: datasets.Dataset) -> datasets.Dataset:
        if self.columns is not None:
            ds = ds.select_columns(self.columns)
        ds.save_to_disk(self.path)
        return ds


class ConcatDatasets:
    def __init__(self, axis=1) -> None:
        self.axis = axis

    def __call__(self, *dses: datasets.Dataset) -> datasets.Dataset:
        ds = datasets.concatenate_datasets(list(dses), axis=self.axis)
        return ds

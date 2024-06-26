"""This module implements a source NER labelling step of the XLNER pipeline"""

from typing import Any
from datasets import Dataset
from transformers import AutoModelForTokenClassification

from src.utils.model_context import use_hf_pipeline
from src.utils.tokenwise_pipeline import register_pipeline


class NERTransform:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        device: int = 0,
        in_column: str = "src_words",
        out_column: str = "src_entities",
        aggregation_strategy: str = "first",
        wordwise: bool = True,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.in_column = in_column
        self.out_column = out_column
        self.wordwise = wordwise
        self.agg_straregy = aggregation_strategy

        if self.wordwise:
            register_pipeline()

    @staticmethod
    def map_ner_out_to_entity(ner_out: dict[str, Any]) -> dict[str, Any]:
        return {
            "start_idx": ner_out["start"],
            "end_idx": ner_out["end"],
            "label": ner_out["entity_group"],
        }

    def __call__(self, ds: Dataset) -> Dataset:
        with use_hf_pipeline(
            "tokenwise-classification" if self.wordwise else "ner",
            self.model_path,
            pipeline_kwargs={
                "device": self.device,
                "batch_size": self.batch_size,
                "aggregation_strategy": self.agg_straregy,
            },
            AutoModelClass=AutoModelForTokenClassification,
        ) as pipe:

            def label(words):
                ner_out_iter = pipe(words)
                entity_batch = []
                for ner_out in ner_out_iter:
                    entities = list(map(self.map_ner_out_to_entity, ner_out))
                    entity_batch.append(entities)
                return {self.out_column: entity_batch}

            ds = ds.map(
                label,
                input_columns=self.in_column,
                batched=True,
                batch_size=self.batch_size,
            )

            return ds

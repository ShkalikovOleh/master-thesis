"""This module implements a source NER labelling step of the XLNER pipeline"""

from functools import partial
from itertools import groupby
from typing import Any, Iterable
from datasets import Dataset
import torch
from transformers import AutoModelForTokenClassification

from src.utils.model_context import use_hf_model, use_hf_pipeline
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
        class_mapping: dict[str, str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device

        self.in_column = in_column
        self.out_column = out_column

        self.wordwise = wordwise
        self.agg_straregy = aggregation_strategy

        self.class_mapping = class_mapping

        if self.wordwise:
            register_pipeline()

    @staticmethod
    def map_ner_out_to_entity(ner_out: dict[str, Any]) -> dict[str, Any]:
        return {
            "start_idx": ner_out["start"],
            "end_idx": ner_out["end"],
            "label": ner_out["entity_group"],
        }

    def map_classes(
        self, entities: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        for entity in entities:
            label = entity["label"]
            if label in self.class_mapping:
                new_label = self.class_mapping[label]
                if new_label != "O":
                    entity["label"] = new_label
                else:  # skip entity
                    continue
            yield entity

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
                    entities = map(self.map_ner_out_to_entity, ner_out)
                    if self.class_mapping is not None:
                        entities = self.map_classes(entities)
                    entities = list(entities)
                    entity_batch.append(entities)
                return {self.out_column: entity_batch}

            ds = ds.map(
                label,
                input_columns=self.in_column,
                batched=True,
                batch_size=self.batch_size,
            )

            return ds


class SubwordEmbeddingExtractor:
    """Extractor which generates source entities spans and target candidates for
    word-to-word alignment task, i.e. for the given source and target words compute
    subword-level embeddings and spans (on token level) of every word. As an output
    returns embeddings as well as token-wise spans of words.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int,
        device: int = 0,
        in_src_words_column: str = "src_words",
        in_tgt_words_column: str = "tokens",
        out_src_emb_column: str = "src_emb",
        out_tgt_emb_column: str = "tgt_emb",
        out_src_spans_column: str = "src_spans",
        out_tgt_spans_column: str = "tgt_spans",
        emb_layer: int = 8,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device

        self.in_src_words_column = in_src_words_column
        self.in_tgt_words_column = in_tgt_words_column
        self.out_src_emb_column = out_src_emb_column
        self.out_tgt_emb_column = out_tgt_emb_column
        self.out_src_spans_column = out_src_spans_column
        self.out_tgt_spans_column = out_tgt_spans_column

        self.emb_layer = emb_layer

    @staticmethod
    def tokenize(tokenizer, words: list[str]):
        tokens = tokenizer(
            words,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=True,
        )
        return tokens

    @staticmethod
    def clean_emb_and_extract_spans(tokens, embs):
        embeddings = []
        spans = []
        for i in range(tokens["input_ids"].shape[0]):
            emb = embs[i]
            word_ids = tokens.word_ids(i)
            mask = torch.BoolTensor([True] * len(word_ids))

            row_spans = []
            curr_pos = 0
            num_skipped = 0
            for key, group in groupby(word_ids):
                length = len(list(group))
                if key is not None:
                    row_spans.append(
                        (curr_pos - num_skipped, curr_pos + length - num_skipped)
                    )
                else:
                    mask[curr_pos : curr_pos + length] = False
                    num_skipped += 1
                curr_pos += length

            spans.append(row_spans)
            embeddings.append(emb[mask])

        return spans, embeddings

    def extract_subwords_emb_and_spans(
        self, src_words: list[list[str]], tgt_words: list[list[str]], model, tokenizer
    ):
        src_tokens = self.tokenize(tokenizer, src_words)
        tgt_tokens = self.tokenize(tokenizer, tgt_words)

        with torch.inference_mode():
            src_embs = model(**src_tokens)["hidden_states"][self.emb_layer].cpu()
            tgt_embs = model(**tgt_tokens)["hidden_states"][self.emb_layer].cpu()

        src_spans, src_embeddings = self.clean_emb_and_extract_spans(
            src_tokens, src_embs
        )
        tgt_spans, tgt_embeddings = self.clean_emb_and_extract_spans(
            tgt_tokens, tgt_embs
        )

        return {
            self.out_src_spans_column: src_spans,
            self.out_src_emb_column: src_embeddings,
            self.out_tgt_spans_column: tgt_spans,
            self.out_tgt_emb_column: tgt_embeddings,
        }

    def __call__(self, ds: Dataset) -> Dataset:
        with use_hf_model(
            self.model_path, model_kwargs={"output_hidden_states": True}
        ) as (model, tokenizer):
            map_func = partial(
                self.extract_subwords_emb_and_spans,
                model=model,
                tokenizer=tokenizer,
            )

            ds = ds.map(
                map_func,
                input_columns=[self.in_src_words_column, self.in_tgt_words_column],
                batched=True,
                batch_size=self.batch_size,
            )

        return ds

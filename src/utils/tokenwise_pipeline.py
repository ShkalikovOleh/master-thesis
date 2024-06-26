"""This module contains an implementation of the
HF Transformers TokenClassificationPipeline which has as an input already splitted
into words/tokens sentence and returns labels for them
"""

import logging
from typing import Iterable, List, Tuple

import numpy as np
from transformers import AutoModelForTokenClassification, TokenClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.token_classification import AggregationStrategy

logger = logging.getLogger("Pipeline")


class TokenwiseClassificationPipeline(TokenClassificationPipeline):
    """Usual Token-Classification (or NER) pipeline expect that an input
    will be in a form of sentence. But sometimes data are already in
    a form of words/tokens and when we try to merge this words into a sentence
    and then run a pipeline we can face some problems. The root of this problem is that
    detokenization and tokenization are not always reversible, e.g. after merging words
    into sentence punctuation signs added to the last word before them which cause
    incorrect labeling for them after tokenization back to words. This pipeline solves
    this problem, because it excpects to have an input in a form of words and not
    sentences and therefore there is no problem with non reversible tokenization.
    Returns NER entities in exactly the same format as TokenClassificationPipeline,
    except the property that start and end indices corresponds to the indices of the
    input tokens (not characters in the sentence).

    All function below is simple alteration of the functions from the base class
    """

    def preprocess(self, tokens, **preprocess_params):
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = (
            True
            if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0
            else False
        )
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=truncation,
            return_special_tokens_mask=True,
            is_split_into_words=True,
            **tokenizer_params,
        )
        inputs.pop("overflow_to_sample_mapping", None)
        num_chunks = len(inputs["input_ids"])

        for i in range(num_chunks):
            model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            model_inputs["tokens"] = tokens if i == 0 else None
            model_inputs["is_last"] = i == num_chunks - 1
            model_inputs["word_ids"] = inputs.word_ids()

            yield model_inputs

    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        word_ids = model_inputs.pop("word_ids")
        tokens = model_inputs.pop("tokens")
        is_last = model_inputs.pop("is_last")

        output = self.model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "word_ids": word_ids,
            "tokens": tokens,
            "is_last": is_last,
            **model_inputs,
        }

    def postprocess(
        self,
        all_outputs,
        aggregation_strategy=AggregationStrategy.FIRST,
        ignore_labels=None,
    ):
        if ignore_labels is None:
            ignore_labels = ["O"]
        all_entities = []
        for model_outputs in all_outputs:
            logits = model_outputs["logits"][0].numpy()
            input_ids = model_outputs["input_ids"][0]
            word_ids = model_outputs["word_ids"]
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            pre_entities = self.gather_pre_entities(
                input_ids,
                scores,
                word_ids,
                special_tokens_mask,
            )
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in ignore_labels
                and entity.get("entity_group", None) not in ignore_labels
            ]
            all_entities.extend(entities)
        num_chunks = len(all_outputs)
        if num_chunks > 1:
            all_entities = self.aggregate_overlapping_entities(all_entities)
        return all_entities

    def gather_pre_entities(
        self,
        input_ids: np.ndarray,
        scores: np.ndarray,
        word_ids: Iterable[int | None],
        special_tokens_mask: np.ndarray,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for
        aggregation"""
        pre_entities = []
        prev_word_id = None
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            word_id = word_ids[idx]
            is_subword = word_id is not None and word_id == prev_word_id
            prev_word_id = word_id

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": word_id,
                "end": word_id + 1,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to B- to not merge tokens
            bi = "B"
            tag = entity_name
        return bi, tag


def register_pipeline():
    PIPELINE_REGISTRY.register_pipeline(
        "tokenwise-classification",
        pipeline_class=TokenwiseClassificationPipeline,
        pt_model=AutoModelForTokenClassification,
    )


register_pipeline()

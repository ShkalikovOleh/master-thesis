from functools import partial
import logging

import numpy as np
from datasets import Dataset
import torch
from transformers import AutoModelForTokenClassification

from src.utils.model_context import use_hf_model


logger = logging.getLogger("Pipeline")


class NERModelLogitsCandidateEvaluator:

    def __init__(
        self,
        model_path: str,
        batch_size: int,
        device: int = 0,
        out_column: str = "tgt_cand_cost",
        tgt_words_column: str = "tokens",
        tgt_cand_column: str = "tgt_candidates",
        subword_aggr_straregy: str = "first",
        per_class_costs: bool = True,
        temperature: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device

        self.out_column = out_column
        self.tgt_words_column = tgt_words_column
        self.tgt_cand_column = tgt_cand_column
        self.subword_aggr_straregy = subword_aggr_straregy
        self.per_class_costs = per_class_costs
        self.temperature = temperature

    @staticmethod
    def tokenize(tgt_words: list[list[str]], tokenizer):
        truncation = (
            True
            if tokenizer.model_max_length and tokenizer.model_max_length > 0
            else False
        )
        tokens = tokenizer(
            tgt_words,
            return_tensors="pt",
            truncation=truncation,
            padding=True,
            is_split_into_words=True,
        )
        tokens.pop("overflow_to_sample_mapping", None)
        return tokens

    @staticmethod
    def compute_logits(model_inputs, model):
        model_inputs.to(model.device)

        with torch.inference_mode():
            output = model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]

        return logits.cpu().numpy()

    def gather_words_scores(
        self, word_ids: list[int], scores: np.ndarray
    ) -> np.ndarray:
        word_logits = []

        prev_word_id = None
        curr_word_start_idx = None
        for idx in range(len(word_ids)):
            word_id = word_ids[idx]
            if prev_word_id != word_ids[idx]:
                if prev_word_id is not None:
                    match self.subword_aggr_straregy:
                        case "first":
                            word_logits.append(scores[curr_word_start_idx])
                        case "max":
                            word_logits.append(np.nanmax(scores))
                        case "average":
                            word_logits.append(np.nanmean(scores))
                curr_word_start_idx = idx

            prev_word_id = word_id

        return np.asarray(word_logits)

    def evaluate_tgt_candidates(
        self,
        tgt_words_batch: list[list[str]],
        tgt_candidates_batch: list[list[tuple[int, int]]],
        model,
        tokenizer,
    ) -> list[np.ndarray] | list[dict[str, np.ndarray]]:
        model_inputs = self.tokenize(tgt_words_batch, tokenizer)

        logits = self.compute_logits(model_inputs, model)
        logits /= self.temperature
        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores_batch = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        costs_batch = []
        for i, tgt_candidates in enumerate(tgt_candidates_batch):
            word_ids = model_inputs.word_ids(i)

            scores = scores_batch[i]
            word_scores = self.gather_words_scores(word_ids, scores)
            max_word_scores = word_scores.max(axis=1, keepdims=True)
            word_scores /= max_word_scores

            if self.per_class_costs:
                costs = {key: [] for key in self._b_labels}
            else:
                costs = {"all": []}
            for cand in tgt_candidates:
                s, e = cand

                # init with scores of B- labels
                cand_cost = {
                    key: word_scores[s, idx] for key, idx in self._b_labels.items()
                }

                # accumulate scores of I- labels
                for word_idx in range(s + 1, e):
                    for key, label_idx in self._i_labels.items():
                        cand_cost[key] *= word_scores[word_idx, label_idx]

                if self.per_class_costs:
                    for key in self._b_labels:
                        costs[key].append(cand_cost[key])
                else:
                    max_cost = cand_cost[max(cand_cost, key=cand_cost.get)]
                    costs["all"].append(max_cost)

            for key in costs:
                costs[key] = np.asarray(costs[key])
            costs_batch.append(costs)

        return costs_batch

    def run_batch(
        self,
        tgt_words: list[list[str]],
        tgt_candidates: list[list[tuple[int, int]]],
        model,
        tokenizer,
    ) -> dict[str, list[np.ndarray] | list[dict[str, np.ndarray]]]:
        costs = self.evaluate_tgt_candidates(
            tgt_words, tgt_candidates, model, tokenizer
        )
        return {self.out_column: costs}

    def _prepare_label_idx_map(self, model):
        b_labels = {}
        i_labels = {}
        for key, idx in model.config.label2id.items():
            if key.startswith("B-"):
                b_labels[key[2:]] = idx
            elif key.startswith("I-"):
                i_labels[key[2:]] = idx

        broken_keys = set(b_labels) ^ set(i_labels)
        for key in b_labels:
            if key in broken_keys:
                i_labels[key] = b_labels[key]
        for key in i_labels:
            if key in broken_keys:
                b_labels[key] = i_labels[key]

        self._b_labels = b_labels
        self._i_labels = i_labels

    def __call__(self, ds: Dataset) -> Dataset:
        with use_hf_model(
            self.model_path,
            AutoModelClass=AutoModelForTokenClassification,
        ) as (model, tokenizer):
            model.to(self.device)
            self._prepare_label_idx_map(model)

            map_func = partial(
                self.run_batch,
                model=model,
                tokenizer=tokenizer,
            )

            ds = ds.map(
                map_func,
                input_columns=[self.tgt_words_column, self.tgt_cand_column],
                batched=True,
                batch_size=self.batch_size,
            )

        return ds


class FilterCandidates:
    """Filter out all candidates with cost computed by NERModelLogitsCandidateEvaluator
    if it is lower than a specified threshold. Allow us to optimize cost calculations
    which requires heavy computations"""

    def __init__(
        self,
        tgt_cand_column: str = "tgt_candidates",
        tgt_cand_cost_column: str = "tgt_cand_cost",
        per_class_costs: bool = True,
        threshold: float = 0.2,
    ) -> None:
        self.tgt_cand_column = tgt_cand_column
        self.tgt_cand_cost_column = tgt_cand_cost_column
        self.per_class_costs = per_class_costs
        self.threshold = threshold

    def filter(
        self, tgt_candidates: list[tuple[int, int]], cand_costs: dict[str, list[float]]
    ) -> list[tuple[int, int]]:
        if self.per_class_costs:
            costs = []
            for i, _ in enumerate(tgt_candidates):
                cand_cost = [cand_costs[key][i] for key in cand_costs]
                costs.append(max(cand_cost))
        else:
            costs = cand_costs["all"]

        filtered_cands = []
        for tgt_cand, cost in zip(tgt_candidates, costs):
            if cost > self.threshold:
                filtered_cands.append(tgt_cand)

        return filtered_cands

    def __call__(self, ds: Dataset) -> Dataset:
        def map_func(
            tgt_candidates: list[tuple[int, int]], cand_costs: dict[str, list[float]]
        ):
            filtered_cands = self.filter(tgt_candidates, cand_costs)
            return {self.tgt_cand_column: filtered_cands}

        ds = ds.map(
            map_func,
            input_columns=[self.tgt_cand_column, self.tgt_cand_cost_column],
            batched=False,
        )

        return ds

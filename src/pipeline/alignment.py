"""This module implements different approaches to compute word-to-word alignments"""

import gc
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Callable, Iterable, Tuple

from datasets import Dataset
import awesome_align.modeling
import awesome_align.tokenization_bert
import numpy as np
import simalign
import torch
from torch.nn.utils.rnn import pad_sequence

from src.ilp.ranges import (
    ProjectionContraint,
    construct_ilp_problem,
    get_cosine_similarity_based_cost,
    greedy_solve_from_costs,
    solve_ilp_problem,
    solve_ilp_with_gurobi,
)


class AlignerBase(ABC):
    def __enter__(self) -> Any:
        pass

    def __exit__(self, type, value, traceback) -> None:
        pass

    @abstractmethod
    def align(
        self, src_words: list[str], tgt_words: list[str]
    ) -> list[Tuple[int, int]]:
        pass

    def align_batched(
        self, src_words_batch: list[list[str]], tgt_words_batch: list[list[str]]
    ) -> Iterable[list[Tuple[int, int]]]:
        for src, tgt in zip(src_words_batch, tgt_words_batch):
            yield self.align(src, tgt)


class WordAlignTransform:
    """Performs word to word alignment calculation as a step of a pipeline.
    Take as an input derived from AlirnerBase class and use it to actual
    computation"""

    def __init__(
        self,
        word_aligner: AlignerBase,
        tgt_column: str,
        src_column: str,
        batch_size: int,
        out_column: str = "word_alignments",
    ) -> None:
        self.word_aligner = word_aligner
        self.tgt_column = tgt_column
        self.src_column = src_column
        self.batch_size = batch_size
        self.out_column = out_column

    def __call__(self, ds: Dataset) -> Dataset:
        with self.word_aligner:

            def align(src_words: list[list[str]], tgt_words: list[list[str]]):
                alignments = list(self.word_aligner.align_batched(src_words, tgt_words))
                return {self.out_column: alignments}

            ds = ds.map(
                align,
                input_columns=[self.src_column, self.tgt_column],
                batched=True,
                batch_size=self.batch_size,
            )
        return ds


class SimAlignAligner(AlignerBase):
    """Computes word to word alignments using SimAlign"""

    def __init__(self, **kwds) -> None:
        super().__init__()
        self.__args = kwds

    def __enter__(self) -> Any:
        self.__aligner = simalign.SentenceAligner(**self.__args)
        return self

    def __exit__(self, type, value, traceback) -> None:
        del self.__aligner
        gc.collect()
        torch.cuda.empty_cache()

    def align(
        self, src_words: list[str], tgt_words: list[str]
    ) -> list[Tuple[int, int]]:
        alignments = self.__aligner.get_word_aligns(src_words, tgt_words)
        for method in alignments:
            return alignments[method]

    def align_batched(
        self, src_words_batch: list[list[str]], tgt_words_batch: list[list[str]]
    ) -> Iterable[list[Tuple[int, int]]]:
        alignments = self.__aligner.get_word_aligns_batched(
            src_words_batch, tgt_words_batch
        )
        return map(lambda align: align[next(iter(align))], alignments)


class AwesomeAligner(AlignerBase):
    """Computes word to word alignments using AWESOME aligner"""

    def __init__(
        self,
        model_path: str,
        extraction: str = "softmax",
        softmax_threshold: float = 0.001,
        align_layer: int = 8,
        device: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.__model_path = model_path
        self.__config = awesome_align.modeling.BertConfig.from_pretrained(
            model_path, cache_dir=cache_dir
        )
        self.__extraction = extraction
        self.__softmax_threshold = softmax_threshold
        self.__align_layer = align_layer
        self.__cache_dir = cache_dir
        self.__device = f"cuda:{device}" if device != -1 else "cpu"

    def __enter__(self) -> Any:
        self.__tokenizer = (
            awesome_align.tokenization_bert.BertTokenizer.from_pretrained(
                self.__model_path, cache_dir=self.__cache_dir
            )
        )
        self.__model = awesome_align.modeling.BertForMaskedLM.from_pretrained(
            self.__model_path,
            from_tf=bool(".ckpt" in self.__model_path),
            config=self.__config,
            cache_dir=self.__cache_dir,
        )
        self.__model.to(self.__device)
        self.__model.eval()
        return self

    def __exit__(self, type, value, traceback) -> None:
        del self.__tokenizer
        del self.__model
        gc.collect()
        torch.cuda.empty_cache()

    def align(
        self, src_words: list[str], tgt_words: list[str]
    ) -> list[Tuple[int, int]]:
        for out in self.align_batched([src_words], [tgt_words]):
            return list(out)  # return the first out

    def tokenize(self, words: list[str]) -> tuple[torch.Tensor, list[int]]:
        tokens = [self.__tokenizer.tokenize(word) for word in words]
        wid = [self.__tokenizer.convert_tokens_to_ids(x) for x in tokens]
        ids = self.__tokenizer.prepare_for_model(
            list(chain(*wid)),
            return_tensors="pt",
            max_length=self.__tokenizer.max_len,
        )["input_ids"][0]

        bpe2word_map = []
        for i, word_list in enumerate(tokens):
            bpe2word_map += [i for x in word_list]

        return ids, bpe2word_map

    def align_batched(
        self, src_words_batch: list[list[str]], tgt_words_batch: list[list[str]]
    ) -> Iterable[list[Tuple[int, int]]]:
        bpe2word_map_src, bpe2word_map_tgt = [], []
        ids_src_list, ids_tgt_list = [], []
        for src_words, tgt_words in zip(src_words_batch, tgt_words_batch):
            src_id, src_b2w = self.tokenize(src_words)
            tgt_id, tgt_b2w = self.tokenize(tgt_words)
            bpe2word_map_src.append(src_b2w)
            bpe2word_map_tgt.append(tgt_b2w)
            ids_src_list.append(src_id)
            ids_tgt_list.append(tgt_id)

        ids_src = pad_sequence(
            ids_src_list, batch_first=True, padding_value=self.__tokenizer.pad_token_id
        ).to(device=self.__device)
        ids_tgt = pad_sequence(
            ids_tgt_list, batch_first=True, padding_value=self.__tokenizer.pad_token_id
        ).to(device=self.__device)

        word_aligns = self.__model.get_aligned_word(
            ids_src,
            ids_tgt,
            bpe2word_map_src,
            bpe2word_map_tgt,
            device=self.__device,
            src_len=0,
            tgt_len=0,
            align_layer=self.__align_layer,
            extraction=self.__extraction,
            softmax_threshold=self.__softmax_threshold,
            test=True,
        )

        for align in word_aligns:
            yield list(align)


class RangeILPAlignTransform:
    """Compute word-to-word alignments by solving range-based ILP problem"""

    def __init__(
        self,
        src_emb_column: str = "src_emb",
        tgt_emb_column: str = "tgt_emb",
        src_spans_column: str = "src_spans",
        tgt_spans_column: str = "tgt_spans",
        threshold: float = 0.7,
        max_reduction: str = "mean",
        out_column: str = "word_alignments",
        n_projected: int = 1,
        proj_constraint: str = "GREATER_OR_EQUAL",
        solver: str = "GUROBI",
        solver_params: dict | None = None,
        num_proc: int | None = None,
    ) -> None:
        assert n_projected >= 0

        self.src_emb_column = src_emb_column
        self.tgt_emb_column = tgt_emb_column
        self.src_spans_column = src_spans_column
        self.tgt_spans_column = tgt_spans_column
        self.out_column = out_column

        self.threshold = threshold
        self.max_reduction = self._get_max_reduction(max_reduction)

        self.n_projected = n_projected
        self.proj_constraint = ProjectionContraint[proj_constraint]
        self.solver = solver

        self.num_proc = num_proc

        if solver_params is None:
            self.solver_params = {}

    @staticmethod
    def _get_max_reduction(reduction: str) -> Callable[[np.ndarray], np.dtype]:
        match reduction:
            case "mean":
                return np.mean
            case "median":
                return np.median
            case "max":
                return np.max

    def align(
        self,
        src_emb: list[list[float]] | np.ndarray,
        tgt_emb: list[list[float]] | np.ndarray,
        src_spans: list[tuple[int, int]],
        tgt_spans: list[tuple[int, int]],
    ) -> list[str]:
        # Calculate matching costs
        costs = get_cosine_similarity_based_cost(
            src_emb,
            tgt_emb,
            src_spans,
            tgt_spans,
            threshold=self.threshold,
            max_reduction=self.max_reduction,
        )

        # Construct and solve ILP problem
        match self.solver:
            case "GREEDY":
                src_idxs, tgt_idxs = greedy_solve_from_costs(
                    costs, tgt_spans, self.n_projected, self.proj_constraint
                )
            case "GUROBI":
                src_idxs, tgt_idxs = solve_ilp_with_gurobi(
                    costs, tgt_spans, self.n_projected, self.proj_constraint
                )
            case _:  # other solvers (via CVXPY)
                n_ent, n_cand = costs.shape
                problem = construct_ilp_problem(
                    costs,
                    tgt_spans,
                    self.n_projected,
                    self.proj_constraint,
                )
                src_idxs, tgt_idxs = solve_ilp_problem(
                    problem, n_ent, n_cand, self.solver, **self.solver_params
                )

        return list(zip(src_idxs, tgt_idxs))

    def __call__(self, ds: Dataset) -> Dataset:
        def align_func(
            src_emb,
            tgt_emb,
            src_spans,
            tgt_spans,
        ):
            word_alignments = self.align(src_emb, tgt_emb, src_spans, tgt_spans)
            return {self.out_column: word_alignments}

        ds = ds.map(
            align_func,
            input_columns=[
                self.src_emb_column,
                self.tgt_emb_column,
                self.src_spans_column,
                self.tgt_spans_column,
            ],
            batched=False,
            num_proc=self.num_proc,
        )

        return ds

"""This module contains different methods to
extract entity candidates in the target sentence
"""

from itertools import chain, combinations
import logging
from typing import Any

from datasets import Dataset

from src.pipeline.projection import (
    HeuriticsProjection,
    BaseProjectionTransform,
)
from src.utils.iterators import min_max


logger = logging.getLogger("Pipeline")


class ContiniousSubrangeExtractor:
    """Returns as candidates all subranges which contains at least
    min_words (default 1) and at max max_words (default number of input words).
    Ignores all subranges that contains a word from the set of stop_words
    """

    def __init__(
        self,
        tgt_words_column: str = "tokens",
        out_column: str = "tgt_candidates",
        min_words: int = 1,
        max_words: int | None = None,
        stop_words: set[str] | list[str] | None = {"!", "?"},
    ) -> None:
        super().__init__()
        assert min_words >= 1

        self.min_words = min_words
        self.max_words = max_words

        self.tgt_words_column = tgt_words_column
        self.out_column = out_column

        if stop_words and isinstance(stop_words, list):
            stop_words = set(stop_words)
        self.stop_words = stop_words

    def generate_subranges(
        self, tgt_words: list[str], start_idx: int, end_idx: int
    ) -> list[tuple[int, int]]:
        max_words = self.max_words if self.max_words else end_idx - start_idx
        max_words += 1
        candidates = []

        for s in range(start_idx, end_idx - self.min_words + 1):
            max_end = min(s + max_words, end_idx + 1)
            for e in range(s + self.min_words, max_end):
                if self.stop_words:
                    if not bool(self.stop_words.intersection(tgt_words[s:e])):
                        candidates.append((s, e))
                else:
                    candidates.append((s, e))

        return candidates

    def extract(self, tgt_words: list[str]) -> list[tuple[int, int]]:
        N = len(tgt_words)
        return self.generate_subranges(tgt_words, 0, N)

    def __call__(self, ds: Dataset) -> Dataset:
        def extract_func(tgt_words):
            candidates = self.extract(tgt_words)
            return {self.out_column: candidates}

        ds = ds.map(extract_func, input_columns=self.tgt_words_column, batched=False)
        return ds


class AlignedContiniousSubrangeExtractor(ContiniousSubrangeExtractor):
    """Returns as candidates all subranges of ranges where the first and the last words
    are aligned to source entities. The lenght of the surbranges can be controlled
    by min_words (default 1) and max_words (default number of input words) parameters.
    Ignores all subranges that contains a word from the set of stop_words.
    The main idea of this extractor is to shrink candidate set by removing all
    candidates which have no chance to be matched.
    """

    def __init__(
        self,
        tgt_words_column: str = "tokens",
        out_column: str = "tgt_candidates",
        src_entities_column: str = "src_entities",
        alignments_column: str = "word_alignments",
        min_words: int = 1,
        max_words: int | None = None,
        stop_words: set[str] | list[str] | None = {"!", "?"},
    ) -> None:
        super().__init__(tgt_words_column, out_column, min_words, max_words, stop_words)

        self.alignment_column = alignments_column
        self.src_entities_column = src_entities_column

    def extract(
        self,
        tgt_words: list[str],
        src_entities: list[dict[str, Any]],
        alignments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(src_entities) == 0:
            return []

        candidates = set()

        n_src_words = max(src_entities, key=lambda ent: ent["end_idx"])["end_idx"]
        aligned_to_src_words = BaseProjectionTransform.gather_aligned_words(
            alignments, n_src_words, to_tgt=False
        )

        for src_entity in src_entities:
            s_idx, e_idx = src_entity["start_idx"], src_entity["end_idx"]

            tgt_aligned_idxs = chain.from_iterable(aligned_to_src_words[s_idx:e_idx])
            min_cand_idx, max_cand_idx = min_max(tgt_aligned_idxs)
            if min_cand_idx is None:  # no aligned tgt words
                continue

            entity_cands = self.generate_subranges(
                tgt_words, min_cand_idx, max_cand_idx + 1
            )
            for cand in entity_cands:
                candidates.add(cand)

        return list(candidates)

    def __call__(self, ds: Dataset) -> Dataset:
        def extract_func(tgt_words, src_entities, alignments):
            candidates = self.extract(tgt_words, src_entities, alignments)
            return {self.out_column: candidates}

        ds = ds.map(
            extract_func,
            input_columns=[
                self.tgt_words_column,
                self.src_entities_column,
                self.alignment_column,
            ],
            batched=False,
        )
        return ds


class AlignedSubrangeMergingExtractor:
    """Consider as candidates only continious ranges of aligned words (ignoring
    their subranges) and their concatenation. Continiuous subranges are generated
    with use the same logic as candidates in the heuristic-based projection approach"""

    def __init__(
        self,
        tgt_words_column: str = "tokens",
        out_column: str = "tgt_candidates",
        src_entities_column: str = "src_entities",
        alignments_column: str = "word_alignments",
        add_all_candidates_subranges: bool = True,
    ) -> None:
        self.tgt_words_column = tgt_words_column
        self.out_column = out_column
        self.alignment_column = alignments_column
        self.src_entities_column = src_entities_column
        self.add_all_candidates_subranges = add_all_candidates_subranges

    def extract(
        self,
        tgt_words: list[str],
        src_entities: list[dict[str, Any]],
        alignments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(src_entities) == 0:
            return []

        candidates = set()

        n_src_words = max(src_entities, key=lambda ent: ent["end_idx"])["end_idx"]
        aligned_by_src_words = BaseProjectionTransform.gather_aligned_words(
            alignments, n_src_words, to_tgt=False
        )

        for src_entity in src_entities:
            s_idx, e_idx = src_entity["start_idx"], src_entity["end_idx"]

            entity_cands = HeuriticsProjection.generate_candidates_from_alignment(
                len(tgt_words), s_idx, e_idx, aligned_by_src_words
            )

            # add candidates itself
            for cand in entity_cands:
                candidates.add(cand)
                if self.add_all_candidates_subranges:
                    start_idx, end_idx = cand
                    for i in range(start_idx, end_idx):
                        for j in range(i + 1, end_idx + 1):
                            candidates.add((i, j))

            # add merged candidates
            for cand1, cand2 in combinations(entity_cands, r=2):
                merged_cand = (cand1[0], cand2[1])
                candidates.add(merged_cand)

        return list(candidates)

    def __call__(self, ds: Dataset) -> Dataset:
        def extract_func(tgt_words, src_entities, alignments):
            candidates = self.extract(tgt_words, src_entities, alignments)
            return {self.out_column: candidates}

        ds = ds.map(
            extract_func,
            input_columns=[
                self.tgt_words_column,
                self.src_entities_column,
                self.alignment_column,
            ],
            batched=False,
        )
        return ds

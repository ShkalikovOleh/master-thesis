"""This module contains different methods to
extract entity candidates in the target sentence
"""

import logging

from datasets import Dataset


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

    def extract(self, tgt_words: list[str]) -> list[tuple[int, int]]:
        N = len(tgt_words)
        max_words = self.max_words if self.max_words else N
        max_words += 1
        candidates = []

        for s in range(N - self.min_words + 1):
            max_end = min(s + max_words, N + 1)
            for e in range(s + self.min_words, max_end):
                if self.stop_words:
                    if not bool(self.stop_words.intersection(tgt_words[s:e])):
                        candidates.append((s, e))
                else:
                    candidates.append((s, e))

        return candidates

    def __call__(self, ds: Dataset) -> Dataset:
        def extract_func(tgt_words):
            candidates = self.extract(tgt_words)
            return {self.out_column: candidates}

        ds = ds.map(extract_func, input_columns=self.tgt_words_column, batched=False)
        return ds

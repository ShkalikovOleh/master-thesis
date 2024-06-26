"""This module implements pipeline steps for splitting
sentences into words and merging them back which is required for
working with translations"""

from abc import ABC, abstractmethod

import nltk
from nltk.tokenize import word_tokenize
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer, WhitespaceSplit

from datasets import Dataset


class WordSplitterBase(ABC):
    @abstractmethod
    def split_words(self, sentence: str) -> list[str]:
        pass


class HFSplitterBase(WordSplitterBase):
    def __init__(self, pretokenizer: PreTokenizer) -> None:
        super().__init__()
        self.pretokenizer = pretokenizer

    def split_words(self, sentence: str) -> list[str]:
        return list(map(lambda w: w[0], self.pretokenizer.pre_tokenize_str(sentence)))


class WhitespaceSplitter(HFSplitterBase):
    def __init__(self) -> None:
        super().__init__(WhitespaceSplit())


class HFBertSplitter(HFSplitterBase):
    def __init__(self) -> None:
        super().__init__(BertPreTokenizer())


class NLTKSplitter(WordSplitterBase):
    def __init__(self) -> None:
        nltk.download("punkt", quiet=True)

    def split_words(self, sentence: str) -> list[str]:
        return word_tokenize(sentence)


class SplitSentence:
    def __init__(
        self,
        word_splitter: WordSplitterBase,
        in_column: str,
        out_column: str = "src_words",
    ) -> None:
        self.word_spltter = word_splitter
        self.in_column = in_column
        self.out_column = out_column

    def _apply_splitter(self, sentence: str) -> dict[str, list[str]]:
        words = self.word_spltter.split_words(sentence)
        return {self.out_column: words}

    def __call__(self, ds: Dataset) -> Dataset:
        ds = ds.map(self._apply_splitter, input_columns=self.in_column)
        return ds


class WordMergerBase(ABC):
    @abstractmethod
    def merge_words(self, words: list[str]) -> str:
        pass


class WhitespaceMerger(WordMergerBase):
    def merge_words(self, words: list[str]) -> str:
        return " ".join(words)


class NLTKMerger(WordMergerBase):
    def __init__(self) -> None:
        super().__init__()
        self.detokenizer = nltk.TreebankWordDetokenizer()

    def merge_words(self, words: list[str]) -> str:
        return self.detokenizer.detokenize(words)


class MergeSentence:
    def __init__(
        self,
        word_merger: WordMergerBase,
        in_column: str = "tokens",
        out_column: str = "sentence",
    ) -> None:
        self.word_merger = word_merger
        self.in_column = in_column
        self.out_column = out_column

    def _apply_merger(self, sentence: str) -> dict[str, list[str]]:
        words = self.word_merger.merge_words(sentence)
        return {self.out_column: words}

    def __call__(self, ds: Dataset) -> Dataset:
        ds = ds.map(self._apply_merger, input_columns=self.in_column)
        return ds

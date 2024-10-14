import gc
from itertools import product
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from nmtscore import NMTScorer
from nltk.tokenize import TreebankWordDetokenizer
import torch


def compute_cosine_similarity_based_cost(
    src_emb: list[list[float]] | np.ndarray,
    tgt_emb: list[list[float]] | np.ndarray,
    src_spans: list[tuple[int, int]],
    tgt_spans: list[tuple[int, int]],
    threshold: float = 0.65,
    max_reduction: Callable[[np.ndarray], np.dtype] = np.mean,
) -> csr_matrix:
    """Compute matching costs between words based on cosine similarity between
    token-level embeddings. The calculation is performed in the following manner:
    for every pair of words it takes all token-level embeddings, for every source/
    target word token the maximum cosine similarity is computed among all target/source
    tokens. Them it reduce all maximum for word with a given reduction function.
    The cost of matching between words is computed by taking maximum of cost for target
    and source words and substracting a given threshold.

    Args:
        src_emb (list[list[float]] | np.ndarray): token-level source words embeddings
        tgt_emb (list[list[float]] | np.ndarray): token-level target words embeddings
        src_spans (list[tuple[int, int]]): token-level spans of source words
        tgt_spans (list[tuple[int, int]]): token-level spans of target words
        threshold (float, optional): minimum word-level similarity score to consider
            word can be aligned. Defaults to 0.8.
        max_reduction (Callable[[np.ndarray], np.dtype], optional): Operation to reduce
            maximums of cosine similarities among all tokens of the word.
            Defaults to np.mean.

    Returns:
        csr_matrix: matrix of costs between source words and target words
    """
    cos_dist = cdist(src_emb, tgt_emb, "cosine")
    cos_sim = 1 - cos_dist

    costs = np.empty((len(src_spans), len(tgt_spans)))
    for (i, src_span), (j, tgt_span) in product(
        enumerate(src_spans), enumerate(tgt_spans)
    ):
        sim_st = cos_sim[src_span[0] : src_span[1], tgt_span[0] : tgt_span[1]]

        cs = max_reduction(sim_st.max(axis=1))
        ct = max_reduction(sim_st.max(axis=0))
        cost = max(cs, ct) - threshold

        costs[i, j] = cost

    return csr_matrix(costs)


def compute_alignment_cost(
    alignments_by_src_words: list[list[int]],
    src_entity_spans: list[tuple[int, int]],
    tgt_candidates: list[tuple[int, int]],
) -> csr_matrix:
    """Compute matching costs between source entities and target candidates
    using word to word alignments. This implementation computes scrores as
    a number of aligned word pairs between a source entity and candidate
    divided by sum of number of words in a source entity and a candidate.
    The main idea is that if all entity/candidates words are aligned that
    it is probably a correct matching

    Args:
        alignments_by_src_words (list[list[int]]): list of list of word aligned to
            source words
        src_entity_spans (list[tuple[int, int]]): spans (in word indices) of
            source entities
        tgt_candidates (list[tuple[int, int]]): spans of candidates

    Returns:
        csr_matrix: matrix of costs between source entities and candidates.
            We use csr format because matrix is usually sparse.
    """

    weights = []
    col_indices = []
    row_ptr = [0]

    for src_s, src_e in src_entity_spans:
        src_aligns = alignments_by_src_words[src_s:src_e]
        src_ent_len = src_e - src_s
        for col, (tgt_s, tgt_e) in enumerate(tgt_candidates):
            num_aligned_words = 0
            for tgt_idxs in src_aligns:
                num_aligned_words += sum(
                    1 if tgt_s <= idx < tgt_e else 0 for idx in tgt_idxs
                )

            if num_aligned_words > 0:
                tgt_cand_len = tgt_e - tgt_s
                w = 2 * num_aligned_words / (src_ent_len + tgt_cand_len)

                weights.append(w)
                col_indices.append(col)

        row_ptr.append(len(col_indices))

    mat_shape = (len(src_entity_spans), len(tgt_candidates))
    return csr_matrix((weights, col_indices, row_ptr), shape=mat_shape)


def compute_ner_model_cost(
    src_entities: list[dict[str, Any]],
    tgt_candidates: list[tuple[int, int]],
    scores_per_classes: dict[str, list[float]],
    use_only_score_spans: bool = False,
    threshold: float = 10e-3,
) -> csr_matrix:
    """Compute matching costs between source entities and target candidates
    using scores from NER model

    Args:
        src_entities (list[tuple[int, int]]): source entities
        tgt_candidates (list[tuple[int, int]]): spans of candidates
        scores_per_classes (dict[str, list[float]]): scores obtained from
            NERModelLogitsCandidateEvaluator which represent NER model evaluation
            of the given candidate belong to class
        use_only_score_spans (bool): if true requires that score only evaluate how
            likely candidate span is a true entity span. Defaulf: False.
        threshold (float): if score is lower than threshold the cost will be zero

    Returns:
        csr_matrix: matrix of costs between source entities and candidates.
            We use csr format because matrix is usually sparse.
    """

    weights = []
    col_indices = []
    row_ptr = [0]

    for entity in src_entities:
        label = entity["label"]
        for col, _ in enumerate(tgt_candidates):
            if not use_only_score_spans:
                score = scores_per_classes[label][col]
            else:
                score = scores_per_classes["all"][col]

            if score > threshold:
                weights.append(score)
                col_indices.append(col)

        row_ptr.append(len(col_indices))

    mat_shape = (len(src_entities), len(tgt_candidates))
    return csr_matrix((weights, col_indices, row_ptr), shape=mat_shape)


class NMTScoreCostEvaluator:

    def __init__(
        self,
        nmt_model: str = "nllb-200-distilled-600M",
        device: str = "cuda:0",
        batch_size: int = 8,
        tgt_lang="eng_Latn",
        normalize: bool = True,
        both_directions: bool = True,
        threshold: float = 10e-1,
    ) -> None:
        self.nmt_model = nmt_model
        self.device = device
        self.batch_size = batch_size

        self.tgt_lang = tgt_lang
        self.normalize = normalize
        self.both_directions = both_directions
        self.threshold = threshold

    def __enter__(self):
        self.scorer = NMTScorer(self.nmt_model, device=self.device)
        self.detokenizer = TreebankWordDetokenizer()

    def __exit__(self, exc_type, exc_value, traceback):
        del self.scorer
        gc.collect()
        torch.cuda.empty_cache()

    def extract_phrases(
        self, words: list[str], spans: list[tuple[int, int]]
    ) -> list[str]:
        phrases = []
        for start, end in spans:
            phrase = self.detokenizer.detokenize(words[start:end])
            phrases.append(phrase)
        return phrases

    def __call__(
        self,
        src_words: list[str],
        tgt_words: list[list[str]],
        src_entity_spans: list[tuple[int, int]],
        tgt_candidates: list[tuple[int, int]],
    ) -> csr_matrix:
        src_phrases = self.extract_phrases(src_words, src_entity_spans)
        tgt_phrases = self.extract_phrases(tgt_words, tgt_candidates)

        input_a = []
        input_b = []
        for src_phrase, tgt_phrase in product(src_phrases, tgt_phrases):
            input_a.append(src_phrase)
            input_b.append(tgt_phrase)

        scores = self.scorer.score_cross_likelihood(
            input_a,
            input_b,
            tgt_lang=self.tgt_lang,
            normalize=self.normalize,
            both_directions=self.both_directions,
            translate_kwargs={"batch_size": self.batch_size},
            score_kwargs={"batch_size": self.batch_size},
        )

        weights = []
        col_indices = []
        row_ptr = [0]
        N_ent = len(src_entity_spans)
        N_cand = len(tgt_candidates)
        for row in range(N_ent):
            for col in range(N_cand):
                score = scores[row * N_cand + col]
                if score > self.threshold:
                    weights.append(score)
                    col_indices.append(col)
            row_ptr.append(len(col_indices))

        return csr_matrix((weights, col_indices, row_ptr), shape=(N_ent, N_cand))

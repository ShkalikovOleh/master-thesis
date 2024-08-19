"""This module contains implementations of the annotation projection
steps of the XLNER pipeline. Labels produced by these steps are returned
in IOB2 format."""

from itertools import groupby
import logging
from typing import Any
from datasets import Dataset
from abc import ABC

from src.ilp.ranges import (
    ProjectionContraint,
    get_relative_lenght_cost,
    construct_ilp_problem,
    greedy_solve_from_costs,
    remove_candidates_with_zero_costs,
    remove_entities_with_zero_cost,
    solve_ilp_problem,
    solve_ilp_with_gurobi,
)
from src.utils.entities import get_entities_spans


logger = logging.getLogger("Pipeline")


class Word2WordAlignmentsBasedProjection(ABC):
    def __init__(
        self,
        tgt_words_column: str = "tokens",
        src_words_column: str = "src_words",
        src_entities_column: str = "src_entities",
        alignments_column: str = "word_alignments",
        out_column: str = "labels",
    ) -> None:
        self.tgt_words_column = tgt_words_column
        self.src_words_column = src_words_column
        self.src_entities_column = src_entities_column
        self.alignments_column = alignments_column
        self.out_column = out_column

    @staticmethod
    def sort_entities_by_len(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(entities, key=lambda ent: ent["end_idx"] - ent["start_idx"])

    @staticmethod
    def gather_aligned_words(
        alignments: list[tuple[int, int]], n_words: int, to_tgt: bool = False
    ) -> list[list[int]]:
        """Make a list of tgt/src aligned words that are aligned to every src/tgt words.

        Args:
            alignments (list[tuple[int, int]]): word-to-word alignments between src and
                tgt words
            n_words (int): number of tgt/src words (depends on to_tgt parameter)
            to_tgt (bool, optional): Whether return list for target words.
                Defaults to False.

        Returns:
            list[list[int]]: list where on every i position there is a list of
                indices of tgt/src words that are aligned to i-th src/tgt word
        """

        def key_func(a):
            return a[1 if to_tgt else 0]

        alignments_by_words = [[] for _ in range(n_words)]
        alignments = sorted(alignments, key=key_func)
        for k, g in groupby(alignments, key=key_func):
            if k >= n_words:
                break
            indices = map(lambda a: a[0 if to_tgt else 1], g)
            alignments_by_words[k].extend(indices)
        return alignments_by_words


class HeuriticsProjection(Word2WordAlignmentsBasedProjection):
    """Based on word-to-word alignments project source entities to target
    sentence using heuristics"""

    def __init__(
        self,
        tgt_words_column: str = "tokens",
        src_words_column: str = "src_words",
        src_entities_column: str = "src_entities",
        alignments_column: str = "word_alignments",
        out_column: str = "labels",
        project_top_k: int | None = 1,
        length_ratio_threshold: float | None = None,
        merge_distance: int = 1,
        merge_only_i_labels: bool = True,
    ) -> None:
        super().__init__(
            tgt_words_column,
            src_words_column,
            src_entities_column,
            alignments_column,
            out_column,
        )
        self.length_ratio_threshold = length_ratio_threshold
        self.merge_distance = merge_distance
        self.merge_only_i_labels = merge_only_i_labels
        self.project_top_k = project_top_k

    @staticmethod
    def generate_candidates_from_alignment(
        n_tgt_words: int,
        start_idx: int,
        end_idx: int,
        alignments_by_src_words: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Generate entity candidates in the target sentence. As entity candidate we
        consider any continious range of target words which are aligned to any word
        in the source sentence

        Args:
            n_tgt_words (int): number of target words
            start_idx (int): index of a first word of a source entity
            end_idx (int): index of a last word of a source entity
            alignments_by_src_words (list[list[int]]): list of list of indices where
                every i-th list correspond to indices of target words that are
                aligned to i-th src words

        Returns:
            list[tuple[int, int]]: list of candidates as a list of indices of start and
                end words
        """

        aligned_mask = [0] * n_tgt_words
        for i in range(start_idx, end_idx):
            for idx in alignments_by_src_words[i]:
                aligned_mask[idx] = 1

        candidates = []
        curr_pos = 0
        for k, g in groupby(aligned_mask):
            lenght = len(list(g))
            if k == 1:
                candidates.append((curr_pos, curr_pos + lenght))
            curr_pos += lenght

        return candidates

    @staticmethod
    def merge_adjacent_candidates(
        candidates: list[tuple[int, int]],
        max_distance: int = 1,
        merge_only_i_labels: bool = True,
        aligned_to_b_labels: list[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Merge candidates if a distance between them less than threshold.
        Useful to handle gaps in alignemnts

        Args:
            candidates (list[tuple[int, int]]): possible target projected entities
            max_distance (int, optional): Maximum distance between candidates to be
                merged. Defaults to 1.
            merge_only_i_labels (bool, optional): whether merge candidates only if they
                starts with word aligned to only I labels of source entity.
                Defaults to True.
            aligned_to_b_labels (list[int] | None, optional): Indices of words that are
                aligned to word with B label. Used only if merge_only_i_labels is True.
                Defaults to None.

        Returns:
            list[tuple[int, int]]: list of merged candidate in a form of tuple of
            indices of the start and end words (zero-indexed)
        """

        candidates = sorted(candidates, key=lambda cand: cand[0])

        merged_candidates = [candidates[0]]
        for cand in candidates[1:]:
            if cand[0] - merged_candidates[-1][1] > max_distance:
                merged_candidates.append(cand)
            elif merge_only_i_labels and cand[0] in aligned_to_b_labels:
                # beggining of cand is aligned to B- word
                merged_candidates.append(cand)
            else:  # extend previous candidate
                prev_start = merged_candidates[-1][0]
                merged_candidates[-1] = (prev_start, cand[1])

        return merged_candidates

    def project(
        self,
        tgt_words: list[str],
        src_words: list[str],
        src_entities: list[dict[str, Any]],
        alignments: list[tuple[int, int]],
    ) -> list[str]:
        # automatically solve annotation collision and substring issues
        entities = self.sort_entities_by_len(src_entities)
        alignments_by_src_words = self.gather_aligned_words(
            alignments, len(src_words), to_tgt=False
        )
        n_tgt_words = len(tgt_words)

        labels = labels = ["O"] * n_tgt_words  # initially no entities
        for entity in entities:
            start_idx, end_idx = entity["start_idx"], entity["end_idx"]
            label = entity["label"]
            len_entity = end_idx - start_idx

            # Extract projection candidates
            candidates = self.generate_candidates_from_alignment(
                n_tgt_words, start_idx, end_idx, alignments_by_src_words
            )

            # Handle split annotations via merging
            if self.merge_distance > 0 and len(candidates) > 1:
                candidates = self.merge_adjacent_candidates(
                    candidates,
                    max_distance=self.merge_distance,
                    merge_only_i_labels=self.merge_only_i_labels,
                    aligned_to_b_labels=alignments_by_src_words[start_idx],
                )

            # Filter out small (ratio of lenght between source entity and candidate is
            # below the threshold) candidates which caused by wrong alignments
            if self.length_ratio_threshold is not None:
                candidates = filter(
                    lambda c: (c[1] - c[0]) / len_entity > self.length_ratio_threshold,
                    candidates,
                )

            # Project source entity only to top k candidates
            if self.project_top_k is not None:
                projected_entities = sorted(
                    candidates, key=lambda cand: cand[1] - cand[0]
                )
                top_k = min(len(projected_entities), self.project_top_k)
                candidates = projected_entities[:top_k]

            # Labelling
            for ent_start, ent_end in candidates:
                if any(map(lambda label: label != "O", labels[ent_start:ent_end])):
                    # skip candidate if it overlaps already projected entity
                    continue
                labels[ent_start] = "B-" + label
                for i in range(ent_start + 1, ent_end):
                    labels[i] = "I-" + label

        return labels

    def __call__(self, ds: Dataset) -> Dataset:
        def project_func(tgt_words, src_words, src_entities, alignments):
            labels = self.project(tgt_words, src_words, src_entities, alignments)
            return {self.out_column: labels}

        ds = ds.map(
            project_func,
            input_columns=[
                self.tgt_words_column,
                self.src_words_column,
                self.src_entities_column,
                self.alignments_column,
            ],
            batched=False,
        )

        return ds


class RangeILPProjection(Word2WordAlignmentsBasedProjection):
    """Projection by solving weighted bipartite matching ILP problem"""

    def __init__(
        self,
        tgt_words_column: str = "tokens",
        src_words_column: str = "src_words",
        src_entities_column: str = "src_entities",
        alignments_column: str = "word_alignments",
        tgt_cand_column: str = "tgt_candidates",
        out_column: str = "labels",
        n_projected: int = 1,
        proj_constraint: str = "EQUAL",
        solver: str = "GUROBI",
        solver_params: dict | None = None,
        num_proc: int | None = None,
        remove_entities_with_zero_cost: bool = True,
        remove_candidates_with_zero_cost: bool = True,
    ) -> None:
        assert n_projected >= 0

        super().__init__(
            tgt_words_column,
            src_words_column,
            src_entities_column,
            alignments_column,
            out_column,
        )
        self.n_projected = n_projected
        self.tgt_cand_column = tgt_cand_column
        self.proj_constraint = ProjectionContraint[proj_constraint]
        self.solver = solver
        self.num_proc = num_proc
        self.remove_entities_with_zero_cost = remove_entities_with_zero_cost
        self.remove_candidates_with_zero_cost = remove_candidates_with_zero_cost

        if solver_params is None:
            self.solver_params = {}

    def project(
        self,
        tgt_words: list[str],
        src_words: list[str],
        src_entities: list[dict[str, Any]],
        alignments: list[tuple[int, int]],
        tgt_candidates: list[tuple[int, int]],
    ) -> list[str]:
        labels = ["O"] * len(tgt_words)
        if len(src_entities) == 0 or len(tgt_candidates) == 0:
            return labels, 0, 0

        aligns_by_src_words = self.gather_aligned_words(
            alignments, len(src_words), to_tgt=False
        )
        src_entities_spans = get_entities_spans(src_entities)

        # Calculate matching costs
        costs = get_relative_lenght_cost(
            aligns_by_src_words, src_entities_spans, tgt_candidates
        )

        # Don't consider entites and candidates that have zero costs
        if self.remove_entities_with_zero_cost:
            costs, nnz_rows = remove_entities_with_zero_cost(costs)

        if self.remove_candidates_with_zero_cost:
            costs, nnz_cols = remove_candidates_with_zero_costs(costs)
            tgt_candidates = list(map(tgt_candidates.__getitem__, nnz_cols))

        # Construct and solve ILP problem
        match self.solver:
            case "GREEDY":
                ent_inds, cand_inds, total_cost = greedy_solve_from_costs(
                    costs, tgt_candidates, self.n_projected, self.proj_constraint
                )
            case "GUROBI":
                ent_inds, cand_inds, total_cost = solve_ilp_with_gurobi(
                    costs, tgt_candidates, self.n_projected, self.proj_constraint
                )
            case _:  # other solvers (via CVXPY)
                n_ent, n_cand = costs.shape
                problem = construct_ilp_problem(
                    costs,
                    tgt_candidates,
                    self.n_projected,
                    self.proj_constraint,
                )
                ent_inds, cand_inds, total_cost = solve_ilp_problem(
                    problem, n_ent, n_cand, self.solver, **self.solver_params
                )
        rel_cost = total_cost / costs.shape[0]

        if (
            len(ent_inds) < len(src_entities)
            and self.proj_constraint
            not in [ProjectionContraint.LESS, ProjectionContraint.LESS_OR_EQUAL]
            and self.n_projected != 0
        ):
            logger.warn("Not every entity has been matched")

        # Labelling
        if remove_entities_with_zero_cost:
            ent_inds = nnz_rows[ent_inds]

        for r, c in zip(ent_inds, cand_inds):
            label = src_entities[r]["label"]
            tgt_s, tgt_e = tgt_candidates[c]

            labels[tgt_s] = "B-" + label
            for idx in range(tgt_s + 1, tgt_e):
                labels[idx] = "I-" + label

        return labels, total_cost, rel_cost

    def __call__(self, ds: Dataset) -> Dataset:
        def project_func(
            tgt_words, src_words, src_entities, alignments, tgt_candidates
        ):
            labels, total_cost, rel_cost = self.project(
                tgt_words, src_words, src_entities, alignments, tgt_candidates
            )
            return {
                self.out_column: labels,
                "total_cost": total_cost,
                "rel_cost": rel_cost,
            }

        ds = ds.map(
            project_func,
            input_columns=[
                self.tgt_words_column,
                self.src_words_column,
                self.src_entities_column,
                self.alignments_column,
                self.tgt_cand_column,
            ],
            batched=False,
            num_proc=self.num_proc,
        )

        return ds

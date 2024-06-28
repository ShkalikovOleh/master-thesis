"""This modules contains utilities to work with target candidates and source entities"""

from itertools import combinations
from typing import Any


def get_entities_spans(entities: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(e["start_idx"], e["end_idx"]) for e in entities]


def get_overlapped_candidates_idxs(
    tgt_candidates: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    max_pos = max(tgt_candidates, key=lambda span: span[1])[1]

    overlapped = set()
    # for every position store all candidates thah hit it
    pos_hit_list = [[] for _ in range(max_pos)]

    for cand_idx, (start_idx, end_idx) in enumerate(tgt_candidates):
        for i in range(start_idx, end_idx):
            pos_hit_list[i].append(cand_idx)

    for pos_hits in pos_hit_list:
        if len(pos_hits) > 1:  # several cand hit this position
            for a, b in combinations(pos_hits, 2):
                overlapped.add((a, b))

    return list(overlapped)

"""Utilities for running pipelines"""

from collections import namedtuple
from typing import Any
from omegaconf import DictConfig

from hydra.utils import instantiate

import networkx as nx

from src.pipeline.data import LoadDataset


Transform = namedtuple("Transform", ["name", "deps", "transform"])


def instantiate_transforms(transforms_cfg: DictConfig) -> list[Transform]:
    result = []
    for name, transform in transforms_cfg.items():
        trans = instantiate(transform["transform"])
        result.append(Transform(name, transform["deps"], trans))
    return result


def shrink_cached(
    graph: nx.DiGraph,
    transforms: list[Transform],
    cached_step_outs: dict[str, Any],
) -> list[Transform]:
    # Find all steps that already included in cache
    cached_deps = set()
    for node in cached_step_outs:
        cached_deps.update(nx.ancestors(graph, node))

    changed = True
    while changed:
        changed = False
        non_cached = set()
        for cand in cached_deps:
            for edge in graph.edges(cand):
                ancestor = edge[1]
                # there is an edge to non cached step
                if ancestor not in cached_deps and ancestor not in cached_step_outs:
                    # can not remove step since there is a deps to non cached transform
                    non_cached.add(cand)
                    changed = True
        for prev_node in non_cached:
            cached_deps.remove(prev_node)

    shrinked_transforms = []
    idx = 0
    for transform in transforms:
        node = transform.name
        if node in cached_step_outs:
            pipe_step = LoadDataset(cached_step_outs[node])
            new_transform = Transform(node, [], pipe_step)
            shrinked_transforms.append(new_transform)
            graph.nodes[node]["idx"] = idx
            idx += 1
        elif node not in cached_deps:
            shrinked_transforms.append(transform)
            graph.nodes[node]["idx"] = idx
            idx += 1
        else:
            graph.remove_node(node)

    return graph, shrinked_transforms


def create_pipeline_graph(transforms: list[Transform]) -> nx.DiGraph:
    graph = nx.DiGraph()

    # add all nodes
    for idx, transform in enumerate(transforms):
        v = transform.name
        graph.add_node(v, idx=idx)

    # add edges
    for transform in transforms:
        v = transform.name
        for u in transform.deps:
            graph.add_edge(u, v)

    return graph

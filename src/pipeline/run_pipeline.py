"""This module gives ability to run a pipeline with specified congif from the
configs directory"""

import logging
from collections import namedtuple

import hydra
import networkx as nx
from omegaconf import DictConfig, OmegaConf


from src.utils.pipeline import (
    create_pipeline_graph,
    instantiate_transforms,
    shrink_cached,
)

Transform = namedtuple("Transform", ["name", "deps", "transform"])

logger = logging.getLogger("Pipeline")


def execute_pipeline(
    transforms: list[Transform],
    cached_step_paths: DictConfig,
) -> None:
    graph = create_pipeline_graph(transforms)
    if not nx.is_directed_acyclic_graph(graph):
        logger.critical("Pipeline has cycle(s)!")
        raise ValueError("Pipeline graph is not a DAG! Please fix a config")

    if cached_step_paths is not None:
        logging.info("Shrink pipeline and reuse cached outputs")
        cache = OmegaConf.to_container(cached_step_paths, resolve=True)
        graph, transforms = shrink_cached(graph, transforms, cache)

    step_outs = {}
    for name in nx.topological_sort(graph):
        step = transforms[graph.nodes[name]["idx"]]
        logger.info(f"Start {name} pipeline step")

        inputs = [step_outs[node] for node in step.deps]
        out = step.transform(*inputs)
        step_outs[name] = out


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.info("Instantiate pipeline and runner")
    transforms = instantiate_transforms(cfg.pipeline)

    if cfg["log_to_wandb"]:
        import wandb

        run = wandb.init(
            project=cfg["wandb_project"], config=OmegaConf.to_container(cfg)
        )
        with run:
            logger.info(f"Start wandb logging into run: {run.name} with id {run.id}")

            execute_pipeline(
                transforms,
                cfg["cached_step_paths"],
            )

            out_dir = hydra.runtime.output_dir
            run.log_artifact(f"{out_dir}/.hydra/config.yaml")
            run.log_artifact(f"{out_dir}/.hydra/overrides.yaml")
    else:
        execute_pipeline(transforms, cfg["cached_step_paths"])


if __name__ == "__main__":
    main()

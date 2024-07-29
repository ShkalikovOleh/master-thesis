import argparse
from collections import namedtuple
from itertools import product
import subprocess

Constraint = namedtuple("Constrain", ["type", "n_proj"])


def make_constraints_option_list(cfg: argparse.Namespace) -> list[Constraint]:
    constraints = []

    for n_proj in cfg.n_proj_less:
        constraints.append(Constraint(type="LESS", n_proj=n_proj))
    for n_proj in cfg.n_proj_leq:
        constraints.append(Constraint(type="LESS_OR_EQUAL", n_proj=n_proj))
    for n_proj in cfg.n_proj_eq:
        constraints.append(Constraint(type="EQUAL", n_proj=n_proj))
    for n_proj in cfg.n_proj_greq:
        constraints.append(Constraint(type="GREATER_OR_EQUAL", n_proj=n_proj))
    for n_proj in cfg.n_proj_greater:
        constraints.append(Constraint(type="GREATER", n_proj=n_proj))

    return constraints


def main(cfg: argparse.Namespace):
    constraints = make_constraints_option_list(cfg)

    for lang in cfg.langs:
        orig_ds_params = [
            "pipeline.load_ds.transform.dataset_path=ShkalikovOleh/europarl-ner",
            "pipeline.load_ds.transform.split=test",
            f"pipeline.load_ds.transform.cfg_name={lang}",
        ]
        align_ds_param = f"pipeline.load_alignments.transform.dataset_path={cfg.aligns_path_prefix}_{lang}"  # noqa
        src_ds_param = (
            f"pipeline.load_entities.transform.dataset_path={cfg.src_entities_path}"
        )

        for pipeline, solver, constraint in product(
            cfg.pipelines, cfg.solvers, constraints
        ):
            pipe_param = f"pipeline=partial/ranges/{pipeline}"
            solver_param = f"pipeline.project.transform.solver={solver}"
            constr_type_param = (
                f"pipeline.project.transform.proj_constraint={constraint.type}"
            )
            constr_n_proj_param = (
                f"pipeline.project.transform.n_projected={constraint.n_proj}"
            )

            python_call = [
                "python",
                "-m",
                "src.pipeline.run_pipeline",
                f"tgt_lang={lang}",
                *orig_ds_params,
                align_ds_param,
                src_ds_param,
                pipe_param,
                solver_param,
                constr_type_param,
                constr_n_proj_param,
            ]

            match cfg.executor:
                case "python":
                    subprocess.run(python_call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sctipt to run sets of experiments on Europarl-based NER dataset"
    )

    parser.add_argument(
        "--src-entities-path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--aligns-path-prefix",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="python",
        choices=["python"],
    )
    parser.add_argument(
        "--n-proj-leq",
        nargs="*",
        type=int,
        default=[1],
    )
    parser.add_argument(
        "--n-proj-less",
        nargs="*",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--n-proj-eq",
        nargs="*",
        type=int,
        default=[1],
    )
    parser.add_argument(
        "--n-proj-greq",
        nargs="*",
        type=int,
        default=[0, 1],
    )
    parser.add_argument(
        "--n-proj-greater",
        nargs="*",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        type=str,
        choices=["de", "es", "it"],
        default=["de", "es", "it"],
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        type=str,
        choices=["GUROBI", "GREEDY"],
        default=["GUROBI", "GREEDY"],
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        choices=["ngrams", "aligned_ngrams", "aligned_subranges"],
        default=["aligned_ngrams", "aligned_subranges"],
    )

    cfg = parser.parse_args()

    main(cfg)

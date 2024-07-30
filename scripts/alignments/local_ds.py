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

    for words_path, gold_path, src_lang, tgt_lang in zip(
        cfg.parallel_text_paths, cfg.gold_aligns_paths, cfg.src_langs, cfg.tgt_langs
    ):
        words_ds_param = f"pipeline.load_ds.transform.path={words_path}"
        gold_ds_param = f"pipeline.load_gold.transform.path={gold_path}"
        tgt_lang_param = f"tgt_lang={tgt_lang}"
        src_lang_param = f"src_lang={src_lang}"
        problem_param = "problem=alignments"

        calls = []

        pipe_param = "pipeline=alignments/full/3rd_party_aligner"
        for aligner in cfg.other_aligners:
            aligner_param = f"aligner={aligner}"

            python_call = [
                "python",
                "-m",
                "src.pipeline.run_pipeline",
                problem_param,
                words_ds_param,
                gold_ds_param,
                tgt_lang_param,
                src_lang_param,
                pipe_param,
                aligner_param,
            ]

            calls.append(python_call)

        pipe_param = "pipeline=alignments/full/ranges_ilp"
        for solver, constraint, reduction, threshold in product(
            cfg.solvers, constraints, cfg.max_reductions, cfg.thresholds
        ):
            solver_param = f"pipeline.align.transform.solver={solver}"
            constr_type_param = (
                f"pipeline.align.transform.proj_constraint={constraint.type}"
            )
            constr_n_proj_param = (
                f"pipeline.align.transform.n_projected={constraint.n_proj}"
            )
            reduction_param = f"pipeline.align.transform.max_reduction={reduction}"
            threshold_param = f"pipeline.align.transform.threshold={threshold}"

            python_call = [
                "python",
                "-m",
                "src.pipeline.run_pipeline",
                problem_param,
                words_ds_param,
                gold_ds_param,
                tgt_lang_param,
                src_lang_param,
                pipe_param,
                solver_param,
                constr_type_param,
                constr_n_proj_param,
                reduction_param,
                threshold_param,
            ]

            calls.append(python_call)

        match cfg.executor:
            case "python":
                for call in calls:
                    subprocess.run(call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sctipt to run sets of experiments for word-to-word alignment task"
    )

    parser.add_argument(
        "--parallel-text-paths",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--gold-aligns-paths",
        nargs="+",
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
        "--max_reductions",
        nargs="+",
        type=str,
        choices=["max", "mean", "median"],
        default=["max", "mean"],
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7],
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
    parser.add_argument("--src-langs", nargs="+", type=str, required=True)
    parser.add_argument("--tgt-langs", nargs="+", type=str, required=True)
    parser.add_argument(
        "--embed-models",
        nargs="+",
        type=str,
        default=["bert-base-multilingual-cased"],
    )
    parser.add_argument("--emb-layer", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--solvers",
        nargs="+",
        type=str,
        choices=["GUROBI", "GREEDY"],
        default=["GUROBI", "GREEDY"],
    )
    parser.add_argument(
        "--other-aligners",
        nargs="*",
        choices=[
            "simalign_mbert_argmax",
            "simalign_mbert_iterative",
            "simalign_mbert_matching",
        ],
        default=[
            "simalign_mbert_argmax",
            "simalign_mbert_iterative",
            "simalign_mbert_matching",
        ],
    )

    cfg = parser.parse_args()

    main(cfg)

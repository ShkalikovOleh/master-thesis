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


lang_code_map = {"de": "deu_Latn", "it": "ita_Latn", "es": "spa_Latn"}


def main(cfg: argparse.Namespace):
    constraints = make_constraints_option_list(cfg)

    for lang in cfg.langs:
        ds_params = [
            "pipeline.load_ds.transform.dataset_path=ShkalikovOleh/europarl-ner",
            "pipeline.load_ds.transform.split=test",
            f"pipeline.load_ds.transform.cfg_name={lang}",
            f"pipeline.load_entities.transform.dataset_path={cfg.src_entities_path}",
            f"pipeline.load_alignments.transform.dataset_path={cfg.align_path}",
        ]

        for pipeline, solver, constraint in product(
            cfg.pipelines, cfg.solvers, constraints
        ):
            pipe_param = f"pipeline=annotation/partial/ranges/{pipeline}"
            solver_param = f"pipeline.project.transform.solver={solver}"
            constr_type_param = (
                f"pipeline.project.transform.proj_constraint={constraint.type}"
            )
            constr_n_proj_param = (
                f"pipeline.project.transform.n_projected={constraint.n_proj}"
            )

            additional_params = [
                f"++pipeline.cand_extraction.transform.max_words={cfg.max_cand_length}"
            ]
            if pipeline == "ner":
                curr_ds_params = ds_params[:-1]
                additional_params.append(
                    f"pipeline.cand_eval.transform.model_path={cfg.ner_model}"
                )
                additional_params.append(
                    f"pipeline.cand_eval.transform.batch_size={cfg.ner_batch_size}"
                )
            elif pipeline == "nmtscore":
                curr_ds_params = ds_params[:-1]
                additional_params.append(
                    (
                        "pipeline.project.transform.cost_params.0.tgt_lang="
                        f"{lang_code_map[lang]}"
                    )
                )
                additional_params.append(
                    (
                        "pipeline.project.transform.cost_params.0.batch_size="
                        f"{cfg.trans_batch_size}"
                    )
                )
            else:
                curr_ds_params = ds_params

            python_call = [
                "python",
                "-m",
                "src.pipeline.run_pipeline",
                f"tgt_lang={lang}",
                *curr_ds_params,
                pipe_param,
                solver_param,
                constr_type_param,
                constr_n_proj_param,
                *additional_params,
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
        "--align_path",
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
        default=[1, 2],
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
        "--ner-model",
        type=str,
        default="ShkalikovOleh/mdeberta-v3-base-conll2003-en",
    )
    parser.add_argument(
        "--ner-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--trans-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument("--max-cand-length", type=int, required=True)
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
        choices=["ngrams", "aligned_subranges", "ner", "nmtscore"],
        default=["aligned_subranges", "ner", "nmtscore"],
    )

    cfg = parser.parse_args()

    main(cfg)

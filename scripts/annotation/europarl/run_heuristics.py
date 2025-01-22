import argparse
from collections import namedtuple
from functools import partial
from itertools import product
import subprocess

Constraint = namedtuple("Constrain", ["type", "n_proj"])


def make_merge_only_i_labels_params(cfg: argparse.Namespace) -> list[Constraint]:
    params = []
    if cfg.merge_only_i_labels != "true":
        params.append("pipeline.project.transform.merge_only_i_labels=false")
    if cfg.merge_only_i_labels != "false":
        params.append("pipeline.project.transform.merge_only_i_labels=true")
    return params


def main(cfg: argparse.Namespace):
    merge_params = make_merge_only_i_labels_params(cfg)

    pipe_param = "pipeline=annotation/partial/heuristics"
    for lang in cfg.langs:
        orig_ds_params = [
            "pipeline.load_ds.transform.dataset_path=ShkalikovOleh/europarl-ner",
            "pipeline.load_ds.transform.split=test",
            f"pipeline.load_ds.transform.cfg_name={lang}",
        ]
        align_ds_param = (
            f"pipeline.load_alignments.transform.dataset_path={cfg.align_path}"
        )
        src_ds_param = (
            f"pipeline.load_entities.transform.dataset_path={cfg.src_entities_path}"
        )

        for rat_thr, k, merge_dist, merge_param in product(
            cfg.length_ratio_thresholds,
            cfg.project_top_ks,
            cfg.merge_distances,
            merge_params,
        ):
            thr_param = f"pipeline.project.transform.length_ratio_threshold={rat_thr}"
            k_param = f"pipeline.project.transform.project_top_k={k}"
            merge_dist_param = f"pipeline.project.transform.merge_distance={merge_dist}"

            python_call = [
                "python",
                "-m",
                "src.pipeline.run_pipeline",
                f"tgt_lang={lang}",
                *orig_ds_params,
                align_ds_param,
                src_ds_param,
                pipe_param,
                merge_param,
                thr_param,
                k_param,
                merge_dist_param,
            ]

            match cfg.executor:
                case "python":
                    subprocess.run(python_call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sctipt to run sets of experiments on Europarl-based NER dataset"
    )

    def type_or_null(value, type_ctor):
        if value == "None":
            return "null"
        return type_ctor(value)

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
        "--langs",
        nargs="+",
        type=str,
        choices=["de", "es", "it"],
        default=["de", "es", "it"],
    )
    parser.add_argument(
        "--length-ratio-thresholds",
        nargs="*",
        type=partial(type_or_null, type_ctor=float),
        default=["null", 0.8],
    )
    parser.add_argument(
        "--project-top-ks",
        nargs="*",
        type=partial(type_or_null, type_ctor=int),
        default=["null", 1],
    )
    parser.add_argument(
        "--merge-distances",
        nargs="*",
        type=int,
        default=[0, 1],
    )
    parser.add_argument(
        "--merge-only-i-labels",
        type=str,
        choices=["both", "true", "false"],
        default="both",
    )

    cfg = parser.parse_args()

    main(cfg)

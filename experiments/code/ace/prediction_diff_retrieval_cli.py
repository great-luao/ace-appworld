import argparse
import json
import os
from typing import Any

from appworld import update_root
from appworld.common.utils import ensure_package_installed
from appworld_experiments.code.ace.prediction_diff_retrieval import (
    DEFAULT_DATAPOINTS_FILE_NAME,
    DEFAULT_METADATA_FILE_NAME,
    DEFAULT_TOP_K,
    PredictionDiffRetrievalClassifier,
    build_retrieval_datapoints,
    resolve_index_dir,
    resolve_source_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query lightweight prediction-diff retrieval datapoints."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="AppWorld root directory. Defaults to current directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-datapoints",
        help="Build retrieval_datapoints.jsonl under a source experiment output folder.",
    )
    add_source_arguments(build_parser)
    build_parser.add_argument(
        "--dataset",
        dest="dataset_names",
        nargs="+",
        default=None,
        help="One or more AppWorld datasets used to filter source task ids.",
    )
    build_parser.add_argument("--task-id", default=None, help="Build from a single task id.")
    build_parser.add_argument(
        "--output-file-name",
        default=DEFAULT_DATAPOINTS_FILE_NAME,
        help="Datapoints file name written under the retrieval index folder.",
    )
    build_parser.add_argument(
        "--metadata-file-name",
        default=DEFAULT_METADATA_FILE_NAME,
        help="Metadata file name written under the retrieval index folder.",
    )
    build_parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing compatible datapoints file instead of rebuilding it.",
    )

    classify_parser = subparsers.add_parser(
        "classify-one",
        help="Classify one query datapoint from a JSON file.",
    )
    add_source_arguments(classify_parser)
    classify_parser.add_argument(
        "--datapoints-file-name",
        default=DEFAULT_DATAPOINTS_FILE_NAME,
        help="Datapoints file name under the retrieval index folder.",
    )
    classify_parser.add_argument(
        "--query-file",
        required=True,
        help="JSON file containing current_code, predicted_output, and actual_output.",
    )
    classify_parser.add_argument(
        "--backend",
        default="hybrid_tfidf",
        choices=["hybrid_tfidf", "hybrid_bm25", "char_ngram_tfidf", "bm25"],
    )
    classify_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    classify_parser.add_argument("--evidence-top-n", type=int, default=5)
    classify_parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path for the classification JSON result. Defaults to stdout.",
    )
    return parser.parse_args()


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-experiment",
        default="ReAct_non_ACE_evaluation",
        help="Source experiment name under experiments/outputs.",
    )
    parser.add_argument(
        "--source-output-dir",
        default=None,
        help="Explicit source output folder. Overrides --source-experiment when provided.",
    )


def main() -> None:
    args = parse_args()
    update_root(args.root)
    ensure_package_installed("appworld_experiments")

    if args.command == "build-datapoints":
        datapoints, metadata = build_retrieval_datapoints(
            source_experiment_name=args.source_experiment,
            source_output_dir=args.source_output_dir,
            dataset_names=args.dataset_names,
            task_id=args.task_id,
            output_file_name=args.output_file_name,
            metadata_file_name=args.metadata_file_name,
            reuse_existing=args.reuse_existing,
        )
        source_output_dir = resolve_source_output_dir(
            source_experiment_name=args.source_experiment,
            source_output_dir=args.source_output_dir,
        )
        index_dir = resolve_index_dir(source_output_dir)
        print(
            "[prediction_diff_retrieval] "
            f"wrote={os.path.join(index_dir, args.output_file_name)} "
            f"datapoints={len(datapoints)} tasks={metadata['task_count']}"
        )
        return

    if args.command == "classify-one":
        source_output_dir = resolve_source_output_dir(
            source_experiment_name=args.source_experiment,
            source_output_dir=args.source_output_dir,
        )
        index_dir = resolve_index_dir(source_output_dir)
        datapoints_path = os.path.join(index_dir, args.datapoints_file_name)
        query = read_json_file(args.query_file)
        classifier = PredictionDiffRetrievalClassifier.from_datapoints_file(
            datapoints_path=datapoints_path,
            backend_name=args.backend,
            top_k=args.top_k,
            evidence_top_n=args.evidence_top_n,
        )
        result = classifier.classify(query)
        if args.output_file:
            write_json_file(args.output_file, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")


def read_json_file(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(file_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()

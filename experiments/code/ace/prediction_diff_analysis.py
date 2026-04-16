import argparse
import copy
import os

from appworld import update_root
from appworld.common.path_store import path_store
from appworld.common.utils import ensure_package_installed, jsonnet_load
from appworld.task import load_task_ids
from appworld_experiments.code.ace.prediction_diff_classifier import PredictionDiffClassifier


def load_classifier_runner_config(config_name: str) -> dict:
    ensure_package_installed("appworld_experiments")
    config_path = os.path.join(path_store.experiment_configs, config_name + ".jsonnet")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    experiment_config = jsonnet_load(
        config_path,
        APPWORLD_PROJECT_PATH=path_store.root,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    runner_type = experiment_config.pop("type")
    runner_config = experiment_config.pop("config")
    if experiment_config:
        raise ValueError(f"Unexpected keys in experiment config: {experiment_config}")
    if runner_type != "ace":
        raise ValueError(f"Unsupported runner type for prediction diff analysis: {runner_type}")
    if runner_config.get("run_type") != "prediction-diff-classification":
        raise ValueError(
            "prediction diff analysis script only supports configs with "
            "run_type='prediction-diff-classification'."
        )
    return runner_config


def resolve_task_ids(
    runner_config: dict,
    dataset_names: list[str],
    task_id: str | None = None,
) -> list[str]:
    if task_id is not None:
        return [task_id]

    custom_task_ids = runner_config.pop("task_ids", None)
    sample_size = runner_config.pop("sample_size", None)
    if custom_task_ids is not None:
        task_ids = custom_task_ids
    else:
        task_ids = []
        seen_task_ids = set()
        for dataset_name in dataset_names:
            for current_task_id in load_task_ids(dataset_name):
                if current_task_id in seen_task_ids:
                    continue
                seen_task_ids.add(current_task_id)
                task_ids.append(current_task_id)
    if sample_size is not None:
        task_ids = task_ids[:sample_size]
    return task_ids


def build_classifier(agent_config: dict) -> PredictionDiffClassifier:
    config = copy.deepcopy(agent_config)
    config_type = config.pop("type", None)
    if config_type and config_type != "prediction_diff_classifier":
        raise ValueError(
            f"Invalid classifier agent type: {config_type}. "
            "Expected 'prediction_diff_classifier'."
        )
    return PredictionDiffClassifier(**config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing prediction-diff classification outputs into analysis files."
    )
    parser.add_argument("config_name", help="Experiment config name without .jsonnet suffix.")
    parser.add_argument(
        "dataset_names",
        nargs="+",
        help="One or more dataset names used to choose task ids for analysis.",
    )
    parser.add_argument("--task-id", dest="task_id", default=None, help="Analyze only one task.")
    parser.add_argument(
        "--root",
        default=".",
        help="AppWorld root directory. Defaults to the current directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_root(args.root)

    runner_config = load_classifier_runner_config(args.config_name)
    agent_config = runner_config.pop("agent")
    runner_config.pop("run_type")
    runner_config.pop("dataset", None)
    runner_config.pop("num_epochs", None)
    runner_config.pop("skip_existing_outputs", None)
    if runner_config:
        unexpected = {key: value for key, value in runner_config.items() if key not in {"task_ids", "sample_size"}}
        if unexpected:
            raise ValueError(f"Unexpected keys in runner config: {unexpected}")

    task_ids = resolve_task_ids(
        runner_config,
        dataset_names=args.dataset_names,
        task_id=args.task_id,
    )
    classifier = build_classifier(agent_config)
    classifier.analyze_tasks(
        task_ids=task_ids,
        dataset_name="+".join(args.dataset_names),
    )


if __name__ == "__main__":
    main()

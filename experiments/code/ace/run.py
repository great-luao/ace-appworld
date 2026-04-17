import copy
import os
from typing import Any

from appworld.common.path_store import path_store
from appworld.task import Task, load_task_ids
from appworld_experiments.code.ace.adaptation_agent import StarAgent
from appworld_experiments.code.ace.base_agent import BaseAgent
from appworld_experiments.code.ace.evaluation_agent import Agent
from appworld_experiments.code.ace.prediction_diff_classifier import PredictionDiffClassifier
from appworld_experiments.code.ace.prediction_diff_curator import PredictionDiffCurator


def _build_prediction_diff_agent(agent_config: dict[str, Any]) -> PredictionDiffClassifier:
    config = copy.deepcopy(agent_config)
    config_type = config.pop("type", None)
    if config_type and config_type != "prediction_diff_classifier":
        raise ValueError(
            f"Invalid classifier agent type: {config_type}. "
            "Expected 'prediction_diff_classifier'."
        )
    return PredictionDiffClassifier(**config)


def _build_prediction_diff_curator(agent_config: dict[str, Any]) -> PredictionDiffCurator:
    config = copy.deepcopy(agent_config)
    config_type = config.pop("type", None)
    if config_type and config_type != "prediction_diff_curator":
        raise ValueError(
            f"Invalid curator agent type: {config_type}. "
            "Expected 'prediction_diff_curator'."
        )
    return PredictionDiffCurator(**config)


def _filter_existing_output_task_ids(
    task_ids: list[str], experiment_name: str
) -> tuple[list[str], list[str]]:
    filtered_task_ids = []
    skipped_task_ids = []
    for current_task_id in task_ids:
        task_output_directory = os.path.join(
            path_store.experiment_outputs, experiment_name, "tasks", current_task_id
        )
        if os.path.exists(task_output_directory):
            skipped_task_ids.append(current_task_id)
        else:
            filtered_task_ids.append(current_task_id)
    return filtered_task_ids, skipped_task_ids


def _filter_existing_classifier_output_task_ids(
    task_ids: list[str],
    source_experiment_name: str,
) -> tuple[list[str], list[str]]:
    filtered_task_ids = []
    skipped_task_ids = []
    for current_task_id in task_ids:
        output_file_path = os.path.join(
            path_store.experiment_outputs,
            source_experiment_name,
            "tasks",
            current_task_id,
            "logs",
            "prediction_diff_classification.jsonl",
        )
        environment_log_path = os.path.join(
            path_store.experiment_outputs,
            source_experiment_name,
            "tasks",
            current_task_id,
            "logs",
            "environment_io.md",
        )
        if os.path.exists(output_file_path) and _classifier_output_is_complete(
            output_file_path=output_file_path,
            environment_log_path=environment_log_path,
        ):
            skipped_task_ids.append(current_task_id)
        else:
            filtered_task_ids.append(current_task_id)
    return filtered_task_ids, skipped_task_ids


def _classifier_output_is_complete(output_file_path: str, environment_log_path: str) -> bool:
    if not os.path.exists(output_file_path) or not os.path.exists(environment_log_path):
        return False

    with open(output_file_path, "r", encoding="utf-8") as file:
        output_line_count = sum(1 for line in file if line.strip())
    with open(environment_log_path, "r", encoding="utf-8") as file:
        environment_interaction_count = sum(
            1 for line in file if line.startswith("### Environment Interaction")
        )
    return output_line_count == environment_interaction_count and environment_interaction_count > 0


def _resolve_task_ids(
    runner_config: dict[str, Any],
    task_id: str | None = None,
    dataset_name_override: str | None = None,
) -> list[str]:
    dataset_name = runner_config.pop("dataset", None)
    if dataset_name_override is not None:
        dataset_name = dataset_name_override
    sample_size = runner_config.pop("sample_size", None)
    custom_task_ids = runner_config.pop("task_ids", None)
    runner_config.pop("num_epochs", 1)
    runner_config.pop("skip_existing_outputs", False)

    if task_id:
        task_ids = [task_id]
    elif custom_task_ids:
        task_ids = custom_task_ids
        print(f"Using custom task list: {len(task_ids)} tasks")
    else:
        if dataset_name is None:
            raise Exception("Either 'dataset' or 'task_ids' must be specified in the config")
        task_ids = load_task_ids(dataset_name)
        if sample_size is not None:
            task_ids = task_ids[:sample_size]
    return task_ids


def run_experiment(
    experiment_name: str,
    runner_config: dict[str, Any],
    task_id: str | None = None,
    num_processes: int = 1,
    process_index: int | None = 0,
) -> None:
    process_index = process_index or 0
    num_processes = num_processes or 1
    run_type = runner_config.pop("run_type")
    agent_config = runner_config.pop("agent")
    num_epochs = runner_config.pop("num_epochs", 1)
    skip_existing_outputs = runner_config.pop("skip_existing_outputs", False)

    if run_type == "prediction-diff-curation":
        if num_epochs != 1:
            raise ValueError("prediction-diff-curation only supports num_epochs=1.")
        if skip_existing_outputs:
            raise ValueError(
                "prediction-diff-curation does not support skip_existing_outputs because "
                "playbook updates are sequential across tasks."
            )

    task_ids = _resolve_task_ids(runner_config, task_id=task_id)

    if runner_config:
        raise Exception(f"Unexpected keys in the runner config: {runner_config}")

    if skip_existing_outputs:
        skip_reference_experiment_name = experiment_name
        if run_type == "prediction-diff-classification":
            source_experiment_name = agent_config.get("source_experiment_name")
            if not source_experiment_name:
                raise ValueError(
                    "prediction-diff-classification requires source_experiment_name."
                )
            skip_reference_experiment_name = source_experiment_name
            task_ids, skipped_task_ids = _filter_existing_classifier_output_task_ids(
                task_ids,
                source_experiment_name,
            )
        else:
            task_ids, skipped_task_ids = _filter_existing_output_task_ids(task_ids, experiment_name)
        if skipped_task_ids:
            print(
                f"Skipping {len(skipped_task_ids)} task(s) with existing output artifacts "
                f"under experiment '{skip_reference_experiment_name}'."
            )
        if not task_ids:
            print("No tasks remaining after skipping existing outputs.")
            return

    for current_task_id in task_ids:
        Task.load(task_id=current_task_id)

    task_ids = task_ids * num_epochs

    if run_type == "ace-adaptation":
        agent = StarAgent.from_dict(agent_config)
    elif run_type == "ace-evaluation":
        agent = Agent.from_dict(agent_config)
    elif run_type == "non-ace-evaluation":
        agent = BaseAgent.from_dict(agent_config)
    elif run_type == "prediction-diff-classification":
        agent = _build_prediction_diff_agent(agent_config)
    elif run_type == "prediction-diff-curation":
        agent = _build_prediction_diff_curator(agent_config)
    else:
        raise ValueError(f"Unknown run_type: {run_type}")

    agent.solve_tasks(
        task_ids=task_ids,
        experiment_name=experiment_name,
        num_processes=num_processes,
        process_index=process_index,
    )

import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from appworld.common.path_store import path_store
from appworld.common.utils import read_json, write_json, write_jsonl, yield_jsonl
from appworld.task import load_task_ids
from appworld_experiments.code.ace.prediction_diff_reconstruction import (
    reconstruct_interactions_from_logs,
)
from appworld_experiments.code.ace.skillbank import DIFF_CATEGORIES as SKILL_DIFF_CATEGORIES
from appworld_experiments.code.ace.skillbank import PRIMARY_BOARDS as SKILL_PRIMARY_BOARDS
from appworld_experiments.code.ace.text_similarity_retrieval import (
    DEFAULT_TOP_K,
    DIFF_CATEGORY_LABELS,
    OTHER_BOARD_CATEGORY,
    PRIMARY_BOARD_LABELS,
    RETRIEVAL_TAXONOMY_VERSION,
    aggregate_class_scores,
    build_backend,
    build_surface_diff,
    current_utc_timestamp,
    get_base_task_id,
    map_diff_category,
    normalize_multiline_text,
    pick_best_label,
)


DEFAULT_INDEX_DIR_NAME = "retrieval_index"
DEFAULT_DATAPOINTS_FILE_NAME = "datapoints.jsonl"
DEFAULT_METADATA_FILE_NAME = "metadata.json"
SKIPPED_SKILL_DIFF_CATEGORIES = {"", "match"}


def resolve_source_output_dir(
    source_experiment_name: str | None = None,
    source_output_dir: str | None = None,
) -> str:
    if source_output_dir:
        return os.path.abspath(source_output_dir.replace("/", os.sep))
    if not source_experiment_name:
        raise ValueError("Either source_experiment_name or source_output_dir is required.")
    return os.path.join(path_store.experiment_outputs, source_experiment_name)


def resolve_selected_task_ids(
    dataset_names: list[str] | None = None,
    task_id: str | None = None,
) -> set[str] | None:
    if task_id:
        return {task_id}
    if not dataset_names:
        return None

    selected_task_ids: list[str] = []
    seen_task_ids = set()
    for dataset_name in dataset_names:
        for current_task_id in load_task_ids(dataset_name):
            if current_task_id in seen_task_ids:
                continue
            seen_task_ids.add(current_task_id)
            selected_task_ids.append(current_task_id)
    return set(selected_task_ids)


def load_retrieval_datapoints(datapoints_path: str) -> list[dict[str, Any]]:
    return [normalize_datapoint(record, require_labels=True) for record in yield_jsonl(datapoints_path)]


def resolve_index_dir(source_output_dir: str) -> str:
    return os.path.join(source_output_dir, DEFAULT_INDEX_DIR_NAME)


def build_retrieval_datapoints(
    source_experiment_name: str | None = None,
    source_output_dir: str | None = None,
    dataset_names: list[str] | None = None,
    task_id: str | None = None,
    output_file_name: str = DEFAULT_DATAPOINTS_FILE_NAME,
    metadata_file_name: str = DEFAULT_METADATA_FILE_NAME,
    reuse_existing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_source_output_dir = resolve_source_output_dir(
        source_experiment_name=source_experiment_name,
        source_output_dir=source_output_dir,
    )
    selected_task_ids = resolve_selected_task_ids(dataset_names=dataset_names, task_id=task_id)
    index_dir = resolve_index_dir(resolved_source_output_dir)
    datapoints_path = os.path.join(index_dir, output_file_name)
    metadata_path = os.path.join(index_dir, metadata_file_name)

    if reuse_existing and os.path.exists(datapoints_path) and os.path.exists(metadata_path):
        metadata = read_json(metadata_path)
        if metadata.get("taxonomy_version") == RETRIEVAL_TAXONOMY_VERSION:
            return load_retrieval_datapoints(datapoints_path), metadata

    tasks_root = os.path.join(resolved_source_output_dir, "tasks")
    task_ids = [
        task_name
        for task_name in sorted(os.listdir(tasks_root))
        if os.path.isdir(os.path.join(tasks_root, task_name))
    ]
    if selected_task_ids is not None:
        task_ids = [current_task_id for current_task_id in task_ids if current_task_id in selected_task_ids]

    datapoints: list[dict[str, Any]] = []
    dropped_examples: list[dict[str, Any]] = []
    input_source_counter = Counter()
    prediction_source_counter = Counter()
    dropped_reason_counter = Counter()
    dropped_field_counter = Counter()

    for current_task_id in task_ids:
        task_logs_dir = os.path.join(tasks_root, current_task_id, "logs")
        label_path = os.path.join(task_logs_dir, "prediction_diff_classification.jsonl")
        if not os.path.exists(label_path):
            continue

        label_records = {int(record["interaction_index"]): record for record in yield_jsonl(label_path)}
        if not label_records:
            continue

        inputs_path = os.path.join(task_logs_dir, "prediction_diff_classification_inputs.jsonl")
        classifier_inputs = {}
        if os.path.exists(inputs_path):
            classifier_inputs = {
                int(record["interaction_index"]): record for record in yield_jsonl(inputs_path)
            }

        reconstructed_by_index = {}
        if any(index not in classifier_inputs for index in label_records):
            reconstructed_rows = reconstruct_interactions_from_logs(
                task_id=current_task_id,
                source_logs_dir=task_logs_dir,
            )
            reconstructed_by_index = {
                int(record["interaction_index"]): record for record in reconstructed_rows
            }

        for interaction_index in sorted(label_records):
            label_record = label_records[interaction_index]
            input_row = classifier_inputs.get(interaction_index)
            reconstructed_row = reconstructed_by_index.get(interaction_index)

            if input_row is not None:
                current_code = input_row.get("current_code") or ""
                predicted_output = input_row.get("predicted_output") or ""
                actual_output = input_row.get("actual_output") or ""
                input_source = "prediction_diff_classification_inputs"
                prediction_source = "prediction_diff_classification_inputs"
            else:
                current_code = reconstructed_row.get("current_code") if reconstructed_row else ""
                predicted_output = reconstructed_row.get("predicted_output") if reconstructed_row else ""
                actual_output = reconstructed_row.get("actual_output") if reconstructed_row else ""
                input_source = "reconstructed_logs"
                prediction_source = (
                    reconstructed_row.get("prediction_source") if reconstructed_row else "missing_prediction_source"
                )

            datapoint = normalize_datapoint(
                {
                    "task_id": current_task_id,
                    "base_task_id": get_base_task_id(current_task_id),
                    "interaction_index": interaction_index,
                    "current_code": current_code,
                    "predicted_output": predicted_output,
                    "actual_output": actual_output,
                    "primary_board": label_record.get("primary_board") or "",
                    "diff_category": label_record.get("diff_category") or "",
                    "input_source": input_source,
                    "prediction_source": prediction_source,
                },
                require_labels=True,
            )
            missing_fields = validate_datapoint(datapoint)
            if missing_fields:
                for field in missing_fields:
                    dropped_field_counter[field] += 1
                reason = ",".join(sorted(missing_fields))
                dropped_reason_counter[reason] += 1
                if len(dropped_examples) < 20:
                    dropped_examples.append(
                        {
                            "task_id": current_task_id,
                            "interaction_index": interaction_index,
                            "missing_or_invalid_fields": missing_fields,
                            "input_source": input_source,
                            "prediction_source": prediction_source,
                        }
                    )
                continue

            datapoints.append(datapoint)
            input_source_counter[input_source] += 1
            prediction_source_counter[prediction_source] += 1

    metadata = build_datapoints_metadata(
        source_experiment_name=source_experiment_name,
        source_output_dir=resolved_source_output_dir,
        index_dir=index_dir,
        dataset_names=dataset_names,
        task_id=task_id,
        task_ids=task_ids,
        datapoints=datapoints,
        input_source_counter=input_source_counter,
        prediction_source_counter=prediction_source_counter,
        dropped_reason_counter=dropped_reason_counter,
        dropped_field_counter=dropped_field_counter,
        dropped_examples=dropped_examples,
    )

    os.makedirs(index_dir, exist_ok=True)
    write_jsonl(datapoints, datapoints_path, silent=True)
    write_json(metadata, metadata_path, silent=True)
    return datapoints, metadata


def normalize_datapoint(
    raw_datapoint: dict[str, Any],
    require_labels: bool,
) -> dict[str, Any]:
    datapoint = {
        **raw_datapoint,
        "current_code": normalize_multiline_text(
            raw_datapoint.get("current_code") or raw_datapoint.get("code") or ""
        ),
        "predicted_output": normalize_multiline_text(
            raw_datapoint.get("predicted_output")
            or raw_datapoint.get("predicted_observation")
            or ""
        ),
        "actual_output": normalize_multiline_text(
            raw_datapoint.get("actual_output")
            or raw_datapoint.get("actual_observation")
            or ""
        ),
        "taxonomy_version": RETRIEVAL_TAXONOMY_VERSION,
    }
    datapoint["surface_diff"] = normalize_multiline_text(
        raw_datapoint.get("surface_diff") or build_surface_diff(
            datapoint["predicted_output"],
            datapoint["actual_output"],
        )
    )

    if require_labels:
        datapoint["primary_board"] = str(raw_datapoint.get("primary_board") or "").strip().lower()
        datapoint["diff_category"] = map_diff_category(raw_datapoint.get("diff_category") or "")
        if datapoint["primary_board"] == "other":
            datapoint["diff_category"] = OTHER_BOARD_CATEGORY
    return datapoint


def validate_datapoint(datapoint: dict[str, Any]) -> list[str]:
    missing_or_invalid_fields = []
    for field_name in ["current_code", "predicted_output", "actual_output", "primary_board"]:
        if not datapoint.get(field_name):
            missing_or_invalid_fields.append(field_name)

    primary_board = datapoint.get("primary_board")
    diff_category = datapoint.get("diff_category")
    if primary_board and primary_board not in PRIMARY_BOARD_LABELS:
        missing_or_invalid_fields.append("primary_board")
    if primary_board != "other":
        if not diff_category or diff_category not in DIFF_CATEGORY_LABELS:
            missing_or_invalid_fields.append("diff_category")
    return missing_or_invalid_fields


def build_datapoints_metadata(
    source_experiment_name: str | None,
    source_output_dir: str,
    index_dir: str,
    dataset_names: list[str] | None,
    task_id: str | None,
    task_ids: list[str],
    datapoints: list[dict[str, Any]],
    input_source_counter: Counter,
    prediction_source_counter: Counter,
    dropped_reason_counter: Counter,
    dropped_field_counter: Counter,
    dropped_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    non_other_datapoints = [item for item in datapoints if item["primary_board"] != "other"]
    return {
        "generated_at_utc": current_utc_timestamp(),
        "taxonomy_version": RETRIEVAL_TAXONOMY_VERSION,
        "source_experiment_name": source_experiment_name,
        "source_output_dir": source_output_dir,
        "index_dir": index_dir,
        "dataset_names": dataset_names,
        "task_id": task_id,
        "task_count": len(task_ids),
        "datapoint_count": len(datapoints),
        "dropped_datapoint_count": sum(dropped_reason_counter.values()),
        "input_source_counts": dict(input_source_counter),
        "prediction_source_counts": dict(prediction_source_counter),
        "dropped_reason_counts": dict(dropped_reason_counter),
        "dropped_missing_or_invalid_field_counts": dict(dropped_field_counter),
        "dropped_examples": dropped_examples,
        "primary_board_distribution": dict(Counter(item["primary_board"] for item in datapoints)),
        "diff_category_distribution": dict(Counter(item["diff_category"] for item in non_other_datapoints)),
    }


class PredictionDiffRetrievalClassifier:
    def __init__(
        self,
        datapoints: list[dict[str, Any]],
        backend_name: str = "hybrid_tfidf",
        top_k: int = DEFAULT_TOP_K,
        evidence_top_n: int = 5,
    ) -> None:
        self.datapoints = [normalize_datapoint(item, require_labels=True) for item in datapoints]
        self.backend_name = backend_name
        self.top_k = top_k
        self.evidence_top_n = evidence_top_n
        self.backend = build_backend(backend_name)
        self.backend.fit(self.datapoints)
        self.board_to_indices, self.board_to_diff_to_indices = self.build_index_maps(self.datapoints)

    @classmethod
    def from_datapoints_file(
        cls,
        datapoints_path: str,
        backend_name: str = "hybrid_tfidf",
        top_k: int = DEFAULT_TOP_K,
        evidence_top_n: int = 5,
    ) -> "PredictionDiffRetrievalClassifier":
        return cls(
            datapoints=load_retrieval_datapoints(datapoints_path),
            backend_name=backend_name,
            top_k=top_k,
            evidence_top_n=evidence_top_n,
        )

    def classify(self, query_datapoint: dict[str, Any]) -> dict[str, Any]:
        query = normalize_datapoint(query_datapoint, require_labels=False)
        scores = self.backend.score_query(query)
        board_candidate_indices = [
            index for indices in self.board_to_indices.values() for index in indices
        ]
        board_scores = aggregate_class_scores(
            scores=scores,
            candidate_indices=board_candidate_indices,
            train_datapoints=self.datapoints,
            label_key="primary_board",
            top_k=self.top_k,
        )
        predicted_board = pick_best_label(board_scores)

        category_scores = {}
        predicted_diff_category = OTHER_BOARD_CATEGORY
        if predicted_board != "other":
            category_candidate_indices = [
                index
                for indices in self.board_to_diff_to_indices.get(predicted_board, {}).values()
                for index in indices
            ]
            category_scores = aggregate_class_scores(
                scores=scores,
                candidate_indices=category_candidate_indices,
                train_datapoints=self.datapoints,
                label_key="diff_category",
                top_k=self.top_k,
            )
            predicted_diff_category = pick_best_label(category_scores)

        should_retrieve_skill = is_skill_actionable(predicted_board, predicted_diff_category)
        return {
            "taxonomy_version": RETRIEVAL_TAXONOMY_VERSION,
            "backend": self.backend_name,
            "top_k": self.top_k,
            "predicted_board": predicted_board,
            "predicted_diff_category": predicted_diff_category,
            "should_retrieve_skill": should_retrieve_skill,
            "skill_bucket_key": (
                {"primary_board": predicted_board, "diff_category": predicted_diff_category}
                if should_retrieve_skill
                else None
            ),
            "board_scores": board_scores,
            "category_scores": category_scores,
            "top_evidence": self.collect_top_evidence(scores),
        }

    @staticmethod
    def build_index_maps(
        datapoints: list[dict[str, Any]],
    ) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
        board_to_indices: defaultdict[str, list[int]] = defaultdict(list)
        board_to_diff_to_indices: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, item in enumerate(datapoints):
            board_to_indices[item["primary_board"]].append(index)
            if item["primary_board"] != "other" and item["diff_category"]:
                board_to_diff_to_indices[item["primary_board"]][item["diff_category"]].append(index)
        return dict(board_to_indices), {
            board: dict(diff_to_indices) for board, diff_to_indices in board_to_diff_to_indices.items()
        }

    def collect_top_evidence(self, scores: np.ndarray) -> list[dict[str, Any]]:
        if len(scores) == 0:
            return []
        ranked_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        evidence = []
        for index in ranked_indices[: self.evidence_top_n]:
            datapoint = self.datapoints[index]
            evidence.append(
                {
                    "score": float(scores[index]),
                    "task_id": datapoint.get("task_id"),
                    "interaction_index": datapoint.get("interaction_index"),
                    "primary_board": datapoint.get("primary_board"),
                    "diff_category": datapoint.get("diff_category"),
                    "input_source": datapoint.get("input_source"),
                    "prediction_source": datapoint.get("prediction_source"),
                }
            )
        return evidence


def is_skill_actionable(primary_board: str, diff_category: str) -> bool:
    return (
        primary_board in SKILL_PRIMARY_BOARDS
        and diff_category in SKILL_DIFF_CATEGORIES
        and diff_category not in SKIPPED_SKILL_DIFF_CATEGORIES
    )

import argparse
import copy
import difflib
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import linear_kernel

from appworld import update_root
from appworld.common.path_store import path_store
from appworld.common.utils import ensure_package_installed, write_json, write_jsonl, yield_jsonl
from appworld_experiments.code.ace.prediction_diff_reconstruction import (
    reconstruct_interactions_from_logs,
)


PRIMARY_BOARD_LABELS = [
    "docs_lookup",
    "auth",
    "read_fetch",
    "local_reasoning",
    "other",
]
DIFF_CATEGORY_LABELS = [
    "match",
    "safe_abstraction",
    "schema_or_name_mismatch",
    "missing_decisive_information",
    "value_or_state_mismatch",
    "wrong_action_or_failure_mode",
]
RETRIEVAL_TAXONOMY_VERSION = "classifier_v2"
OTHER_BOARD_CATEGORY = ""
HYBRID_FIELD_WEIGHTS = {
    "predicted_output": 0.35,
    "actual_output": 0.35,
    "current_code": 0.20,
    "surface_diff": 0.10,
}
DEFAULT_BACKENDS = ["hybrid_tfidf", "hybrid_bm25", "char_ngram_tfidf", "bm25"]
DEFAULT_TOP_K = 5
DEFAULT_SPLIT_SEED = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline hierarchical retrieval experiments on prediction diff logs."
    )
    parser.add_argument(
        "--source-experiment",
        default="ReAct_non_ACE_evaluation",
        help="Source experiment name under experiments/outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/outputs/Text_Similarity_Hierarchical_Retrieval",
        help="Output directory for structured artifacts.",
    )
    parser.add_argument(
        "--stage",
        choices=["reconstruct", "full_bank", "prototype_bank", "all"],
        default="all",
        help="Which phase to execute.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        help="Backends for full-bank evaluation.",
    )
    parser.add_argument(
        "--prototype-backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        help="Backends for prototype-bank evaluation.",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--prototype-k-max", type=int, default=20)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use only the first 8 base task ids for a quick end-to-end validation run.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="AppWorld root directory. Defaults to current directory.",
    )
    return parser.parse_args()


def current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_multiline_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def map_diff_category(raw_diff_category: str) -> str:
    return str(raw_diff_category or "").strip().lower()


def get_base_task_id(task_id: str) -> str:
    return task_id.split("_", 1)[0]


def deterministic_word_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def ensure_text(text: str, empty_token: str = "__empty__") -> str:
    value = normalize_multiline_text(text)
    return value if value else empty_token


def build_surface_diff(predicted_output: str, actual_output: str) -> str:
    predicted_lines = normalize_multiline_text(predicted_output).splitlines()
    actual_lines = normalize_multiline_text(actual_output).splitlines()
    diff_lines = list(
        difflib.unified_diff(
            predicted_lines,
            actual_lines,
            fromfile="predicted_output",
            tofile="actual_output",
            lineterm="",
        )
    )
    sections = [
        "PREDICTED_OUTPUT",
        ensure_text(predicted_output, empty_token="(empty predicted output)"),
        "",
        "ACTUAL_OUTPUT",
        ensure_text(actual_output, empty_token="(empty actual output)"),
        "",
        "TEXT_DIFF",
        "\n".join(diff_lines) if diff_lines else "(no line-level diff)",
    ]
    return "\n".join(sections)


def format_char_ngram_text(datapoint: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "CURRENT_CODE",
            ensure_text(datapoint["current_code"]),
            "PREDICTED_OUTPUT",
            ensure_text(datapoint["predicted_output"]),
            "ACTUAL_OUTPUT",
            ensure_text(datapoint["actual_output"]),
            "SURFACE_DIFF",
            ensure_text(datapoint["surface_diff"]),
        ]
    )


def load_or_reconstruct_datapoints(
    source_experiment_name: str,
    output_dir: str,
    allowed_base_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datapoints_path = os.path.join(output_dir, "datapoints.jsonl")
    reconstruction_stats_path = os.path.join(output_dir, "reconstruction_stats.json")
    if os.path.exists(datapoints_path) and os.path.exists(reconstruction_stats_path):
        reconstruction_stats = load_json(reconstruction_stats_path)
        cached_selected_base_task_ids = reconstruction_stats.get("selected_base_task_ids")
        requested_selected_base_task_ids = (
            sorted(allowed_base_task_ids) if allowed_base_task_ids is not None else None
        )
        if (
            reconstruction_stats.get("source_experiment_name") == source_experiment_name
            and reconstruction_stats.get("taxonomy_version") == RETRIEVAL_TAXONOMY_VERSION
            and cached_selected_base_task_ids == requested_selected_base_task_ids
        ):
            datapoints = list(yield_jsonl(datapoints_path))
            return datapoints, reconstruction_stats

    source_root = os.path.join(path_store.experiment_outputs, source_experiment_name, "tasks")
    task_ids = sorted(
        task_name for task_name in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, task_name))
    )

    datapoints: list[dict[str, Any]] = []
    dropped_examples: list[dict[str, Any]] = []
    input_source_counter = Counter()
    prediction_source_counter = Counter()
    alignment_strategy_counter = Counter()
    dropped_reason_counter = Counter()
    dropped_field_counter = Counter()

    stats: dict[str, Any] = {
        "generated_at_utc": current_utc_timestamp(),
        "source_experiment_name": source_experiment_name,
        "taxonomy_version": RETRIEVAL_TAXONOMY_VERSION,
        "task_count": len(task_ids),
        "task_count_with_labels": 0,
        "task_count_with_classifier_inputs": 0,
        "task_count_using_reconstructed_logs": 0,
        "label_record_count": 0,
        "datapoint_count": 0,
        "dropped_datapoint_count": 0,
    }

    for task_id in task_ids:
        base_task_id = get_base_task_id(task_id)
        if allowed_base_task_ids is not None and base_task_id not in allowed_base_task_ids:
            continue

        logs_dir = os.path.join(source_root, task_id, "logs")
        label_path = os.path.join(logs_dir, "prediction_diff_classification.jsonl")
        if not os.path.exists(label_path):
            continue

        label_records = {record["interaction_index"]: record for record in yield_jsonl(label_path)}
        if not label_records:
            continue
        stats["task_count_with_labels"] += 1
        stats["label_record_count"] += len(label_records)

        inputs_path = os.path.join(logs_dir, "prediction_diff_classification_inputs.jsonl")
        classifier_inputs = {}
        if os.path.exists(inputs_path):
            classifier_inputs = {
                record["interaction_index"]: record for record in yield_jsonl(inputs_path)
            }
            stats["task_count_with_classifier_inputs"] += 1

        reconstructed_by_index = {}
        needs_reconstruction = any(
            interaction_index not in classifier_inputs for interaction_index in label_records
        )
        if needs_reconstruction:
            stats["task_count_using_reconstructed_logs"] += 1
            reconstructed_rows = reconstruct_interactions_from_logs(task_id=task_id, source_logs_dir=logs_dir)
            reconstructed_by_index = {
                record["interaction_index"]: record for record in reconstructed_rows
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
                alignment = None
            else:
                current_code = reconstructed_row.get("current_code") if reconstructed_row else ""
                predicted_output = reconstructed_row.get("predicted_output") if reconstructed_row else ""
                actual_output = reconstructed_row.get("actual_output") if reconstructed_row else ""
                input_source = "reconstructed_logs"
                prediction_source = (
                    reconstructed_row.get("prediction_source") if reconstructed_row else "missing_prediction_source"
                )
                alignment = reconstructed_row.get("alignment") if reconstructed_row else None

            datapoint = {
                "task_id": task_id,
                "base_task_id": base_task_id,
                "interaction_index": interaction_index,
                "current_code": normalize_multiline_text(current_code),
                "predicted_output": normalize_multiline_text(predicted_output),
                "actual_output": normalize_multiline_text(actual_output),
                "primary_board": str(label_record.get("primary_board") or "").strip().lower(),
                "diff_category": map_diff_category(label_record.get("diff_category") or ""),
                "input_source": input_source,
                "prediction_source": prediction_source,
                "alignment": alignment,
            }
            datapoint["surface_diff"] = build_surface_diff(
                datapoint["predicted_output"],
                datapoint["actual_output"],
            )
            if datapoint["primary_board"] == "other":
                datapoint["diff_category"] = OTHER_BOARD_CATEGORY

            missing_fields = [
                field
                for field in [
                    "current_code",
                    "predicted_output",
                    "actual_output",
                    "primary_board",
                ]
                if not datapoint[field]
            ]
            if datapoint["primary_board"] != "other" and not datapoint["diff_category"]:
                missing_fields.append("diff_category")
            if missing_fields:
                stats["dropped_datapoint_count"] += 1
                for field in missing_fields:
                    dropped_field_counter[field] += 1
                reason = ",".join(sorted(missing_fields))
                dropped_reason_counter[reason] += 1
                if len(dropped_examples) < 20:
                    dropped_examples.append(
                        {
                            "task_id": task_id,
                            "interaction_index": interaction_index,
                            "missing_fields": missing_fields,
                            "input_source": input_source,
                            "prediction_source": prediction_source,
                        }
                    )
                continue

            datapoints.append(datapoint)
            input_source_counter[input_source] += 1
            prediction_source_counter[prediction_source] += 1
            if alignment is not None:
                alignment_strategy_counter[alignment.get("strategy") or "unknown"] += 1

    stats["datapoint_count"] = len(datapoints)
    stats["selected_base_task_ids"] = (
        sorted(allowed_base_task_ids) if allowed_base_task_ids is not None else None
    )
    stats["input_source_counts"] = dict(input_source_counter)
    stats["prediction_source_counts"] = dict(prediction_source_counter)
    stats["alignment_strategy_counts"] = dict(alignment_strategy_counter)
    stats["dropped_reason_counts"] = dict(dropped_reason_counter)
    stats["dropped_missing_field_counts"] = dict(dropped_field_counter)
    stats["dropped_examples"] = dropped_examples
    stats["primary_board_distribution"] = dict(Counter(item["primary_board"] for item in datapoints))
    stats["diff_category_distribution"] = dict(
        Counter(item["diff_category"] for item in datapoints if item["primary_board"] != "other")
    )

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(datapoints, datapoints_path, silent=True)
    write_json(stats, reconstruction_stats_path, silent=True)
    return datapoints, stats


def build_split_manifest(
    datapoints: list[dict[str, Any]],
    output_dir: str,
    split_seed: int,
    val_ratio: float,
) -> dict[str, Any]:
    split_manifest_path = os.path.join(output_dir, "split_manifest.json")
    if os.path.exists(split_manifest_path):
        existing_manifest = load_json(split_manifest_path)
        if (
            existing_manifest.get("split_seed") == split_seed
            and existing_manifest.get("val_ratio") == val_ratio
            and existing_manifest.get("taxonomy_version") == RETRIEVAL_TAXONOMY_VERSION
        ):
            return existing_manifest

    base_task_ids = sorted({item["base_task_id"] for item in datapoints})
    shuffled_base_task_ids = copy.deepcopy(base_task_ids)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled_base_task_ids)

    validation_count = max(1, int(round(len(base_task_ids) * val_ratio)))
    validation_base_task_ids = sorted(shuffled_base_task_ids[-validation_count:])
    train_base_task_ids = sorted(shuffled_base_task_ids[:-validation_count])
    train_base_task_id_set = set(train_base_task_ids)
    validation_base_task_id_set = set(validation_base_task_ids)

    train_datapoints = [item for item in datapoints if item["base_task_id"] in train_base_task_id_set]
    validation_datapoints = [
        item for item in datapoints if item["base_task_id"] in validation_base_task_id_set
    ]

    manifest = {
        "generated_at_utc": current_utc_timestamp(),
        "taxonomy_version": RETRIEVAL_TAXONOMY_VERSION,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "base_task_id_count": len(base_task_ids),
        "train_base_task_ids": train_base_task_ids,
        "validation_base_task_ids": validation_base_task_ids,
        "train_stats": build_split_stats(train_datapoints),
        "validation_stats": build_split_stats(validation_datapoints),
    }
    write_json(manifest, split_manifest_path, silent=True)
    return manifest


def build_split_stats(datapoints: list[dict[str, Any]]) -> dict[str, Any]:
    diff_counter = Counter(item["diff_category"] for item in datapoints if item["primary_board"] != "other")
    other_board_count = sum(item["primary_board"] == "other" for item in datapoints)
    return {
        "base_task_id_count": len({item["base_task_id"] for item in datapoints}),
        "interaction_count": len(datapoints),
        "primary_board_distribution": dict(Counter(item["primary_board"] for item in datapoints)),
        "diff_category_distribution": dict(diff_counter),
        "other_board_ratio": (other_board_count / len(datapoints)) if datapoints else None,
    }


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        import json

        return json.load(file)


class RetrievalBackend:
    def __init__(self, name: str):
        self.name = name
        self.train_datapoints: list[dict[str, Any]] = []

    def fit(self, train_datapoints: list[dict[str, Any]]) -> None:
        self.train_datapoints = train_datapoints

    def score_query(self, query_datapoint: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def pairwise_similarity(self, candidate_indices: list[int]) -> np.ndarray:
        raise NotImplementedError


class HybridTfidfBackend(RetrievalBackend):
    def __init__(self) -> None:
        super().__init__("hybrid_tfidf")
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.bank_matrices: dict[str, Any] = {}

    def fit(self, train_datapoints: list[dict[str, Any]]) -> None:
        super().fit(train_datapoints)
        for field_name in HYBRID_FIELD_WEIGHTS:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            texts = [ensure_text(item[field_name]) for item in train_datapoints]
            self.vectorizers[field_name] = vectorizer
            self.bank_matrices[field_name] = vectorizer.fit_transform(texts)

    def score_query(self, query_datapoint: dict[str, Any]) -> np.ndarray:
        total_scores = np.zeros(len(self.train_datapoints), dtype=float)
        for field_name, weight in HYBRID_FIELD_WEIGHTS.items():
            query_vector = self.vectorizers[field_name].transform([ensure_text(query_datapoint[field_name])])
            field_scores = linear_kernel(query_vector, self.bank_matrices[field_name]).ravel()
            total_scores += weight * field_scores
        return total_scores

    def pairwise_similarity(self, candidate_indices: list[int]) -> np.ndarray:
        total_scores = np.zeros((len(candidate_indices), len(candidate_indices)), dtype=float)
        for field_name, weight in HYBRID_FIELD_WEIGHTS.items():
            field_matrix = self.bank_matrices[field_name][candidate_indices]
            total_scores += weight * linear_kernel(field_matrix, field_matrix)
        return total_scores


class CharNgramTfidfBackend(RetrievalBackend):
    def __init__(self) -> None:
        super().__init__("char_ngram_tfidf")
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.bank_matrix = None

    def fit(self, train_datapoints: list[dict[str, Any]]) -> None:
        super().fit(train_datapoints)
        texts = [format_char_ngram_text(item) for item in train_datapoints]
        self.bank_matrix = self.vectorizer.fit_transform(texts)

    def score_query(self, query_datapoint: dict[str, Any]) -> np.ndarray:
        query_vector = self.vectorizer.transform([format_char_ngram_text(query_datapoint)])
        return linear_kernel(query_vector, self.bank_matrix).ravel()

    def pairwise_similarity(self, candidate_indices: list[int]) -> np.ndarray:
        subset_matrix = self.bank_matrix[candidate_indices]
        return linear_kernel(subset_matrix, subset_matrix)


class BM25Backend(RetrievalBackend):
    def __init__(self) -> None:
        super().__init__("bm25")
        self.bm25 = None
        self.corpus_tokens: list[list[str]] = []
        self.full_pairwise_similarity: np.ndarray | None = None

    def fit(self, train_datapoints: list[dict[str, Any]]) -> None:
        super().fit(train_datapoints)
        from rank_bm25 import BM25Okapi

        self.corpus_tokens = [
            deterministic_word_tokenize(format_char_ngram_text(item)) for item in train_datapoints
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.full_pairwise_similarity = None

    def score_query(self, query_datapoint: dict[str, Any]) -> np.ndarray:
        query_tokens = deterministic_word_tokenize(format_char_ngram_text(query_datapoint))
        return np.asarray(self.bm25.get_scores(query_tokens), dtype=float)

    def pairwise_similarity(self, candidate_indices: list[int]) -> np.ndarray:
        self._ensure_full_pairwise_similarity()
        index_array = np.asarray(candidate_indices, dtype=int)
        return self.full_pairwise_similarity[np.ix_(index_array, index_array)]

    def _ensure_full_pairwise_similarity(self) -> None:
        if self.full_pairwise_similarity is not None:
            return

        num_docs = len(self.corpus_tokens)
        similarity_matrix = np.zeros((num_docs, num_docs), dtype=np.float32)
        for row_index, query_tokens in enumerate(self.corpus_tokens):
            similarity_matrix[row_index] = np.asarray(
                self.bm25.get_scores(query_tokens),
                dtype=np.float32,
            )
        self.full_pairwise_similarity = 0.5 * (similarity_matrix + similarity_matrix.T)


class HybridBM25Backend(RetrievalBackend):
    def __init__(self) -> None:
        super().__init__("hybrid_bm25")
        self.bm25_by_field: dict[str, Any] = {}
        self.corpus_tokens_by_field: dict[str, list[list[str]]] = {}
        self.full_pairwise_similarity: np.ndarray | None = None

    def fit(self, train_datapoints: list[dict[str, Any]]) -> None:
        super().fit(train_datapoints)
        from rank_bm25 import BM25Okapi

        self.bm25_by_field = {}
        self.corpus_tokens_by_field = {}
        for field_name in HYBRID_FIELD_WEIGHTS:
            field_tokens = [
                deterministic_word_tokenize(ensure_text(item[field_name]))
                for item in train_datapoints
            ]
            self.corpus_tokens_by_field[field_name] = field_tokens
            self.bm25_by_field[field_name] = BM25Okapi(field_tokens)
        self.full_pairwise_similarity = None

    def score_query(self, query_datapoint: dict[str, Any]) -> np.ndarray:
        total_scores = np.zeros(len(self.train_datapoints), dtype=float)
        for field_name, weight in HYBRID_FIELD_WEIGHTS.items():
            query_tokens = deterministic_word_tokenize(ensure_text(query_datapoint[field_name]))
            field_scores = np.asarray(
                self.bm25_by_field[field_name].get_scores(query_tokens),
                dtype=float,
            )
            total_scores += weight * field_scores
        return total_scores

    def pairwise_similarity(self, candidate_indices: list[int]) -> np.ndarray:
        self._ensure_full_pairwise_similarity()
        index_array = np.asarray(candidate_indices, dtype=int)
        return self.full_pairwise_similarity[np.ix_(index_array, index_array)]

    def _ensure_full_pairwise_similarity(self) -> None:
        if self.full_pairwise_similarity is not None:
            return

        num_docs = len(self.train_datapoints)
        total_similarity = np.zeros((num_docs, num_docs), dtype=np.float32)
        for field_name, weight in HYBRID_FIELD_WEIGHTS.items():
            field_similarity = np.zeros((num_docs, num_docs), dtype=np.float32)
            field_bm25 = self.bm25_by_field[field_name]
            field_tokens = self.corpus_tokens_by_field[field_name]
            for row_index, query_tokens in enumerate(field_tokens):
                field_similarity[row_index] = np.asarray(
                    field_bm25.get_scores(query_tokens),
                    dtype=np.float32,
                )
            total_similarity += weight * (0.5 * (field_similarity + field_similarity.T))
        self.full_pairwise_similarity = total_similarity


def build_backend(name: str) -> RetrievalBackend:
    if name == "hybrid_tfidf":
        return HybridTfidfBackend()
    if name == "hybrid_bm25":
        return HybridBM25Backend()
    if name == "char_ngram_tfidf":
        return CharNgramTfidfBackend()
    if name == "bm25":
        return BM25Backend()
    raise ValueError(f"Unknown backend: {name}")


def top_k_mean(scores: list[float], k: int) -> float:
    if not scores:
        return float("-inf")
    sorted_scores = sorted(scores, reverse=True)
    used_scores = sorted_scores[: min(k, len(sorted_scores))]
    return float(np.mean(used_scores))


def pick_best_label(score_map: dict[str, dict[str, float]]) -> str:
    ranked = sorted(
        score_map.items(),
        key=lambda item: (-item[1]["top_k_mean"], -item[1]["top_1"], item[0]),
    )
    return ranked[0][0]


def aggregate_class_scores(
    scores: np.ndarray,
    candidate_indices: list[int],
    train_datapoints: list[dict[str, Any]],
    label_key: str,
    top_k: int,
) -> dict[str, dict[str, float]]:
    label_to_scores: defaultdict[str, list[float]] = defaultdict(list)
    for candidate_index in candidate_indices:
        label = train_datapoints[candidate_index][label_key]
        label_to_scores[label].append(float(scores[candidate_index]))

    aggregated = {}
    for label, label_scores in label_to_scores.items():
        aggregated[label] = {
            "top_k_mean": top_k_mean(label_scores, top_k),
            "top_1": max(label_scores),
            "support": len(label_scores),
        }
    return aggregated


def ensure_confusion_matrix_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(path, exist_ok=True)
    return path


def save_confusion_matrix(
    output_dir: str,
    file_stem: str,
    labels: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> str:
    confusion_dir = ensure_confusion_matrix_dir(output_dir)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    path = os.path.join(confusion_dir, f"{file_stem}.json")
    write_json(
        {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
        path,
        silent=True,
    )
    return path


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = []
    for label, p_value, r_value, f1_value, support_value in zip(
        labels, precision, recall, f1, support
    ):
        per_class.append(
            {
                "label": label,
                "precision": float(p_value),
                "recall": float(r_value),
                "f1": float(f1_value),
                "support": int(support_value),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def compute_board_metrics(
    output_dir: str,
    backend_name: str,
    bank_name: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, Any]:
    metrics = compute_classification_metrics(y_true, y_pred, labels)
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["confusion_matrix_path"] = save_confusion_matrix(
        output_dir=output_dir,
        file_stem=f"{bank_name}_{backend_name}_board",
        labels=labels,
        y_true=y_true,
        y_pred=y_pred,
    )
    return metrics


def compute_category_metrics(
    output_dir: str,
    backend_name: str,
    bank_name: str,
    gold_board_labels: list[str],
    y_true: list[str],
    y_pred_end_to_end: list[str],
    y_pred_gold_board: list[str],
) -> dict[str, Any]:
    end_to_end_metrics = compute_classification_metrics(
        y_true, y_pred_end_to_end, DIFF_CATEGORY_LABELS
    )
    end_to_end_metrics["confusion_matrix_path"] = save_confusion_matrix(
        output_dir=output_dir,
        file_stem=f"{bank_name}_{backend_name}_category_end_to_end",
        labels=DIFF_CATEGORY_LABELS,
        y_true=y_true,
        y_pred=y_pred_end_to_end,
    )

    gold_board_metrics = compute_classification_metrics(
        y_true, y_pred_gold_board, DIFF_CATEGORY_LABELS
    )
    gold_board_metrics["confusion_matrix_path"] = save_confusion_matrix(
        output_dir=output_dir,
        file_stem=f"{bank_name}_{backend_name}_category_gold_board",
        labels=DIFF_CATEGORY_LABELS,
        y_true=y_true,
        y_pred=y_pred_gold_board,
    )

    match_precision, match_recall, match_f1, match_support = precision_recall_fscore_support(
        y_true,
        y_pred_end_to_end,
        labels=["match"],
        zero_division=0,
    )
    non_match_count = sum(label != "match" for label in y_true)
    match_count = sum(label == "match" for label in y_true)

    non_match_predicted_as_match = sum(
        gold != "match" and pred == "match"
        for gold, pred in zip(y_true, y_pred_end_to_end)
    )
    match_predicted_as_non_match = sum(
        gold == "match" and pred != "match"
        for gold, pred in zip(y_true, y_pred_end_to_end)
    )
    other_board_count = sum(board == "other" for board in gold_board_labels)

    return {
        "end_to_end": end_to_end_metrics,
        "gold_board": gold_board_metrics,
        "match": {
            "precision": float(match_precision[0]),
            "recall": float(match_recall[0]),
            "f1": float(match_f1[0]),
            "support": int(match_support[0]),
        },
        "board_conditioned_effect": {
            "category_accuracy_when_board_correct": safe_accuracy(
                y_true,
                y_pred_end_to_end,
                mask=None,
            ),
        },
        "non_match_predicted_as_match_rate": (
            non_match_predicted_as_match / non_match_count if non_match_count else None
        ),
        "match_predicted_as_non_match_rate": (
            match_predicted_as_non_match / match_count if match_count else None
        ),
        "other_board_count_excluded_from_category_metrics": other_board_count,
    }


def safe_accuracy(y_true: list[str], y_pred: list[str], mask: list[bool] | None) -> float | None:
    if mask is None:
        return float(accuracy_score(y_true, y_pred)) if y_true else None
    filtered_gold = [gold for gold, keep in zip(y_true, mask) if keep]
    filtered_pred = [pred for pred, keep in zip(y_pred, mask) if keep]
    if not filtered_gold:
        return None
    return float(accuracy_score(filtered_gold, filtered_pred))


def build_train_index_maps(train_datapoints: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    board_to_indices: defaultdict[str, list[int]] = defaultdict(list)
    board_to_diff_to_indices: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, item in enumerate(train_datapoints):
        board_to_indices[item["primary_board"]].append(index)
        if item["primary_board"] != "other" and item["diff_category"]:
            board_to_diff_to_indices[item["primary_board"]][item["diff_category"]].append(index)
    return dict(board_to_indices), {
        board: dict(diff_to_indices) for board, diff_to_indices in board_to_diff_to_indices.items()
    }


def run_hierarchical_evaluation(
    output_dir: str,
    backend_name: str,
    bank_name: str,
    backend: RetrievalBackend,
    train_datapoints: list[dict[str, Any]],
    validation_datapoints: list[dict[str, Any]],
    board_bank_indices: dict[str, list[int]],
    category_bank_indices: dict[str, dict[str, list[int]]],
    top_k: int,
) -> dict[str, Any]:
    predictions = []
    board_labels = [
        label
        for label in PRIMARY_BOARD_LABELS
        if label in {item["primary_board"] for item in validation_datapoints} or label in board_bank_indices
    ]

    for datapoint in validation_datapoints:
        scores = backend.score_query(datapoint)
        board_score_map = aggregate_class_scores(
            scores=scores,
            candidate_indices=[index for indices in board_bank_indices.values() for index in indices],
            train_datapoints=train_datapoints,
            label_key="primary_board",
            top_k=top_k,
        )
        predicted_board = pick_best_label(board_score_map)

        if predicted_board == "other":
            predicted_category_score_map = {}
            predicted_diff_category = OTHER_BOARD_CATEGORY
        else:
            predicted_board_indices = [
                index
                for indices in category_bank_indices.get(predicted_board, {}).values()
                for index in indices
            ]
            predicted_category_score_map = aggregate_class_scores(
                scores=scores,
                candidate_indices=predicted_board_indices,
                train_datapoints=train_datapoints,
                label_key="diff_category",
                top_k=top_k,
            )
            predicted_diff_category = pick_best_label(predicted_category_score_map)

        if datapoint["primary_board"] == "other":
            gold_board_score_map = {}
            gold_board_diff_category = OTHER_BOARD_CATEGORY
        else:
            gold_board_indices = [
                index
                for indices in category_bank_indices.get(datapoint["primary_board"], {}).values()
                for index in indices
            ]
            gold_board_score_map = aggregate_class_scores(
                scores=scores,
                candidate_indices=gold_board_indices,
                train_datapoints=train_datapoints,
                label_key="diff_category",
                top_k=top_k,
            )
            gold_board_diff_category = pick_best_label(gold_board_score_map)

        predictions.append(
            {
                "task_id": datapoint["task_id"],
                "base_task_id": datapoint["base_task_id"],
                "interaction_index": datapoint["interaction_index"],
                "gold_board": datapoint["primary_board"],
                "predicted_board": predicted_board,
                "gold_diff_category": datapoint["diff_category"],
                "predicted_diff_category_end_to_end": predicted_diff_category,
                "predicted_diff_category_gold_board": gold_board_diff_category,
                "board_scores": board_score_map,
                "predicted_board_category_scores": predicted_category_score_map,
                "gold_board_category_scores": gold_board_score_map,
                "input_source": datapoint["input_source"],
                "prediction_source": datapoint["prediction_source"],
            }
        )

    predictions_path = os.path.join(output_dir, f"predictions_{bank_name}_{backend_name}.jsonl")
    write_jsonl(predictions, predictions_path, silent=True)

    y_true_board = [item["gold_board"] for item in predictions]
    y_pred_board = [item["predicted_board"] for item in predictions]
    category_predictions = [item for item in predictions if item["gold_board"] != "other"]
    y_true_diff = [item["gold_diff_category"] for item in category_predictions]
    y_pred_end_to_end = [item["predicted_diff_category_end_to_end"] for item in category_predictions]
    y_pred_gold_board = [item["predicted_diff_category_gold_board"] for item in category_predictions]

    board_metrics = compute_board_metrics(
        output_dir=output_dir,
        backend_name=backend_name,
        bank_name=bank_name,
        y_true=y_true_board,
        y_pred=y_pred_board,
        labels=board_labels,
    )
    category_metrics = compute_category_metrics(
        output_dir=output_dir,
        backend_name=backend_name,
        bank_name=bank_name,
        gold_board_labels=[item["gold_board"] for item in predictions],
        y_true=y_true_diff,
        y_pred_end_to_end=y_pred_end_to_end,
        y_pred_gold_board=y_pred_gold_board,
    )
    category_board_correct_mask = [
        item["gold_board"] == item["predicted_board"] for item in category_predictions
    ]
    category_metrics["board_conditioned_effect"] = {
        "category_accuracy_when_board_correct": safe_accuracy(
            y_true_diff,
            y_pred_end_to_end,
            category_board_correct_mask,
        ),
        "category_accuracy_when_board_wrong": safe_accuracy(
            y_true_diff,
            y_pred_end_to_end,
            [not keep for keep in category_board_correct_mask],
        ),
    }

    return {
        "generated_at_utc": current_utc_timestamp(),
        "taxonomy_version": RETRIEVAL_TAXONOMY_VERSION,
        "backend": backend_name,
        "bank_name": bank_name,
        "top_k": top_k,
        "train_size": len(train_datapoints),
        "validation_size": len(validation_datapoints),
        "category_evaluated_size": len(category_predictions),
        "board_metrics": board_metrics,
        "category_metrics": category_metrics,
        "predictions_path": predictions_path,
        "example_non_other_to_other_board_errors": collect_prediction_examples(
            predictions,
            predicate=lambda item: item["gold_board"] != "other" and item["predicted_board"] == "other",
        ),
        "example_other_to_non_other_board_errors": collect_prediction_examples(
            predictions,
            predicate=lambda item: item["gold_board"] == "other" and item["predicted_board"] != "other",
        ),
    }


def collect_prediction_examples(
    predictions: list[dict[str, Any]],
    predicate,
    limit: int = 5,
) -> list[dict[str, Any]]:
    examples = []
    for item in predictions:
        if not predicate(item):
            continue
        examples.append(
            {
                "task_id": item["task_id"],
                "interaction_index": item["interaction_index"],
                "gold_board": item["gold_board"],
                "predicted_board": item["predicted_board"],
                "gold_diff_category": item["gold_diff_category"],
                "predicted_diff_category_end_to_end": item["predicted_diff_category_end_to_end"],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def similarity_to_distance(similarity_matrix: np.ndarray) -> np.ndarray:
    if similarity_matrix.size == 0:
        return similarity_matrix
    max_similarity = float(np.max(similarity_matrix))
    return max_similarity - similarity_matrix


def select_k_medoids(similarity_matrix: np.ndarray, k: int, max_iter: int = 20) -> list[int]:
    num_points = similarity_matrix.shape[0]
    if k >= num_points:
        return list(range(num_points))
    distance_matrix = similarity_to_distance(similarity_matrix)

    medoids = [int(np.argmin(distance_matrix.sum(axis=1)))]
    while len(medoids) < k:
        best_candidate = None
        best_cost = None
        for candidate in range(num_points):
            if candidate in medoids:
                continue
            candidate_medoids = medoids + [candidate]
            cost = float(distance_matrix[:, candidate_medoids].min(axis=1).sum())
            if best_cost is None or cost < best_cost - 1e-12:
                best_cost = cost
                best_candidate = candidate
        medoids.append(int(best_candidate))

    for _ in range(max_iter):
        assignments = np.argmin(distance_matrix[:, medoids], axis=1)
        new_medoids: list[int] = []
        selected = set()

        for cluster_index in range(len(medoids)):
            cluster_members = np.where(assignments == cluster_index)[0]
            if len(cluster_members) == 0:
                current_costs = distance_matrix[:, medoids].min(axis=1)
                for candidate in np.argsort(current_costs)[::-1]:
                    if int(candidate) not in selected:
                        cluster_members = np.array([candidate])
                        break
            intra_cluster = distance_matrix[np.ix_(cluster_members, cluster_members)]
            cluster_medoid = int(cluster_members[np.argmin(intra_cluster.sum(axis=1))])
            if cluster_medoid in selected:
                current_costs = distance_matrix[:, medoids].min(axis=1)
                for candidate in np.argsort(current_costs)[::-1]:
                    if int(candidate) not in selected:
                        cluster_medoid = int(candidate)
                        break
            new_medoids.append(cluster_medoid)
            selected.add(cluster_medoid)

        if new_medoids == medoids:
            break
        medoids = new_medoids

    return medoids


def build_prototype_banks(
    train_datapoints: list[dict[str, Any]],
    backend: RetrievalBackend,
    prototype_k_max: int,
) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]], dict[str, Any]]:
    board_to_indices, board_to_diff_to_indices = build_train_index_maps(train_datapoints)
    board_prototype_bank: dict[str, list[int]] = {}
    category_prototype_bank: dict[str, dict[str, list[int]]] = {}
    prototype_stats: dict[str, Any] = {
        "generated_at_utc": current_utc_timestamp(),
        "board_prototypes": {},
        "category_prototypes": {},
    }

    for board, candidate_indices in board_to_indices.items():
        local_similarity = backend.pairwise_similarity(candidate_indices)
        k_value = min(prototype_k_max, math.ceil(math.sqrt(len(candidate_indices))))
        local_medoids = select_k_medoids(local_similarity, k_value)
        selected_indices = [candidate_indices[index] for index in local_medoids]
        board_prototype_bank[board] = selected_indices
        prototype_stats["board_prototypes"][board] = {
            "original_count": len(candidate_indices),
            "prototype_count": len(selected_indices),
            "compression_ratio": len(selected_indices) / len(candidate_indices),
            "selected_datapoints": [
                describe_datapoint_identifier(train_datapoints[index]) for index in selected_indices
            ],
        }

    for board, diff_to_indices in board_to_diff_to_indices.items():
        category_prototype_bank[board] = {}
        prototype_stats["category_prototypes"][board] = {}
        for diff_category, candidate_indices in diff_to_indices.items():
            local_similarity = backend.pairwise_similarity(candidate_indices)
            k_value = min(prototype_k_max, math.ceil(math.sqrt(len(candidate_indices))))
            local_medoids = select_k_medoids(local_similarity, k_value)
            selected_indices = [candidate_indices[index] for index in local_medoids]
            category_prototype_bank[board][diff_category] = selected_indices
            prototype_stats["category_prototypes"][board][diff_category] = {
                "original_count": len(candidate_indices),
                "prototype_count": len(selected_indices),
                "compression_ratio": len(selected_indices) / len(candidate_indices),
                "selected_datapoints": [
                    describe_datapoint_identifier(train_datapoints[index]) for index in selected_indices
                ],
            }

    return board_prototype_bank, category_prototype_bank, prototype_stats


def describe_datapoint_identifier(datapoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": datapoint["task_id"],
        "interaction_index": datapoint["interaction_index"],
        "primary_board": datapoint["primary_board"],
        "diff_category": datapoint["diff_category"],
    }


def write_bank_stats(
    output_dir: str,
    train_datapoints: list[dict[str, Any]],
    validation_datapoints: list[dict[str, Any]],
) -> dict[str, Any]:
    bank_stats = {
        "generated_at_utc": current_utc_timestamp(),
        "train": build_split_stats(train_datapoints),
        "validation": build_split_stats(validation_datapoints),
        "board_to_validation_diff_distribution": build_board_conditioned_diff_distribution(
            validation_datapoints
        ),
        "board_to_train_diff_distribution": build_board_conditioned_diff_distribution(train_datapoints),
    }
    write_json(bank_stats, os.path.join(output_dir, "bank_stats.json"), silent=True)
    return bank_stats


def build_board_conditioned_diff_distribution(datapoints: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    board_to_diff_counter: defaultdict[str, Counter] = defaultdict(Counter)
    for item in datapoints:
        if item["primary_board"] == "other" or not item["diff_category"]:
            continue
        board_to_diff_counter[item["primary_board"]][item["diff_category"]] += 1
    return {board: dict(counter) for board, counter in board_to_diff_counter.items()}


def run_full_bank(
    output_dir: str,
    train_datapoints: list[dict[str, Any]],
    validation_datapoints: list[dict[str, Any]],
    backends: list[str],
    top_k: int,
) -> dict[str, Any]:
    results = {}
    board_bank_indices, category_bank_indices = build_train_index_maps(train_datapoints)
    for backend_name in backends:
        backend = build_backend(backend_name)
        backend.fit(train_datapoints)
        result = run_hierarchical_evaluation(
            output_dir=output_dir,
            backend_name=backend_name,
            bank_name="full_bank",
            backend=backend,
            train_datapoints=train_datapoints,
            validation_datapoints=validation_datapoints,
            board_bank_indices=board_bank_indices,
            category_bank_indices=category_bank_indices,
            top_k=top_k,
        )
        result_path = os.path.join(output_dir, f"results_full_bank_{backend_name}.json")
        write_json(result, result_path, silent=True)
        results[backend_name] = result
    return results


def run_prototype_bank(
    output_dir: str,
    train_datapoints: list[dict[str, Any]],
    validation_datapoints: list[dict[str, Any]],
    backends: list[str],
    top_k: int,
    prototype_k_max: int,
) -> dict[str, Any]:
    results = {}
    for backend_name in backends:
        backend = build_backend(backend_name)
        backend.fit(train_datapoints)
        board_prototype_bank, category_prototype_bank, prototype_stats = build_prototype_banks(
            train_datapoints=train_datapoints,
            backend=backend,
            prototype_k_max=prototype_k_max,
        )
        prototype_stats["backend"] = backend_name
        prototype_stats["prototype_k_max"] = prototype_k_max
        prototype_stats["taxonomy_version"] = RETRIEVAL_TAXONOMY_VERSION
        prototype_stats_path = os.path.join(output_dir, f"prototype_stats_{backend_name}.json")
        write_json(prototype_stats, prototype_stats_path, silent=True)

        result = run_hierarchical_evaluation(
            output_dir=output_dir,
            backend_name=backend_name,
            bank_name="prototype_bank",
            backend=backend,
            train_datapoints=train_datapoints,
            validation_datapoints=validation_datapoints,
            board_bank_indices=board_prototype_bank,
            category_bank_indices=category_prototype_bank,
            top_k=top_k,
        )
        result["prototype_stats_path"] = prototype_stats_path
        result_path = os.path.join(output_dir, f"results_prototype_bank_{backend_name}.json")
        write_json(result, result_path, silent=True)
        results[backend_name] = result
    return results


def collect_existing_results(output_dir: str, prefix: str) -> dict[str, dict[str, Any]]:
    results = {}
    prefix_with_underscore = prefix + "_"
    for file_name in sorted(os.listdir(output_dir)):
        if not file_name.startswith(prefix_with_underscore) or not file_name.endswith(".json"):
            continue
        backend_name = file_name[len(prefix_with_underscore) : -len(".json")]
        path = os.path.join(output_dir, file_name)
        result = load_json(path)
        if result.get("taxonomy_version") != RETRIEVAL_TAXONOMY_VERSION:
            continue
        results[backend_name] = result
    return results


def write_error_analysis(
    output_dir: str,
    reconstruction_stats: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    full_results = collect_existing_results(output_dir, "results_full_bank")
    prototype_results = collect_existing_results(output_dir, "results_prototype_bank")

    lines = [
        "# Text Similarity Hierarchical Retrieval Analysis",
        "",
        f"- generated_at_utc: {current_utc_timestamp()}",
        f"- datapoint_count: {reconstruction_stats.get('datapoint_count', 0)}",
        f"- dropped_datapoint_count: {reconstruction_stats.get('dropped_datapoint_count', 0)}",
        f"- train_interactions: {split_manifest['train_stats']['interaction_count']}",
        f"- validation_interactions: {split_manifest['validation_stats']['interaction_count']}",
        "",
        "## Reconstruction",
        "",
        f"- input_source_counts: {reconstruction_stats.get('input_source_counts', {})}",
        f"- prediction_source_counts: {reconstruction_stats.get('prediction_source_counts', {})}",
        f"- dropped_reason_counts: {reconstruction_stats.get('dropped_reason_counts', {})}",
        "",
    ]

    if full_results:
        lines.extend(
            [
                "## Full-Bank Comparison",
                "",
                "| backend | board_acc | board_macro_f1 | board_bal_acc | e2e_acc | e2e_macro_f1 | gold_board_acc | gold_board_macro_f1 | no_skill_f1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for backend_name, result in sorted(full_results.items()):
            lines.append(
                "| {backend} | {board_acc:.4f} | {board_macro_f1:.4f} | {board_bal_acc:.4f} | {e2e_acc:.4f} | {e2e_macro_f1:.4f} | {gold_acc:.4f} | {gold_macro_f1:.4f} | {no_skill_f1:.4f} |".format(
                    backend=backend_name,
                    board_acc=result["board_metrics"]["accuracy"],
                    board_macro_f1=result["board_metrics"]["macro_f1"],
                    board_bal_acc=result["board_metrics"]["balanced_accuracy"],
                    e2e_acc=result["category_metrics"]["end_to_end"]["accuracy"],
                    e2e_macro_f1=result["category_metrics"]["end_to_end"]["macro_f1"],
                    gold_acc=result["category_metrics"]["gold_board"]["accuracy"],
                    gold_macro_f1=result["category_metrics"]["gold_board"]["macro_f1"],
                    no_skill_f1=result["category_metrics"]["no_skill_needed"]["f1"],
                )
            )
        best_full_backend_name, best_full_result = max(
            full_results.items(),
            key=lambda item: item[1]["category_metrics"]["end_to_end"]["macro_f1"],
        )
        lines.extend(
            [
                "",
                f"- best_full_bank_backend: `{best_full_backend_name}`",
                f"- board_bottleneck_gap: {best_full_result['category_metrics']['gold_board']['accuracy'] - best_full_result['category_metrics']['end_to_end']['accuracy']:.4f}",
                f"- action_required_predicted_as_no_skill_needed_rate: {best_full_result['category_metrics']['action_required_predicted_as_no_skill_needed_rate']}",
                f"- no_skill_needed_predicted_as_action_required_rate: {best_full_result['category_metrics']['no_skill_needed_predicted_as_action_required_rate']}",
                "",
                "### Hardest Classes In Best Full-Bank Backend",
                "",
            ]
        )
        hardest_classes = sorted(
            best_full_result["category_metrics"]["end_to_end"]["per_class"],
            key=lambda item: item["f1"],
        )
        for item in hardest_classes[:6]:
            lines.append(
                "- {label}: f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}, support={support}".format(
                    **item
                )
            )
        lines.extend(
            [
                "",
                "### Action-Required -> No-Skill-Needed Examples",
                "",
            ]
        )
        if best_full_result["example_action_required_to_no_skill_needed_errors"]:
            for example in best_full_result["example_action_required_to_no_skill_needed_errors"]:
                lines.append(
                    "- {task_id}#{interaction_index}: gold={gold_diff_category}, pred={predicted_diff_category_end_to_end}, gold_board={gold_board}, pred_board={predicted_board}".format(
                        **example
                    )
                )
        else:
            lines.append("- none")

    if prototype_results:
        lines.extend(
            [
                "",
                "## Prototype-Bank Comparison",
                "",
                "| backend | e2e_acc | e2e_macro_f1 | gold_board_acc | gold_board_macro_f1 | no_skill_f1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for backend_name, result in sorted(prototype_results.items()):
            lines.append(
                "| {backend} | {e2e_acc:.4f} | {e2e_macro_f1:.4f} | {gold_acc:.4f} | {gold_macro_f1:.4f} | {no_skill_f1:.4f} |".format(
                    backend=backend_name,
                    e2e_acc=result["category_metrics"]["end_to_end"]["accuracy"],
                    e2e_macro_f1=result["category_metrics"]["end_to_end"]["macro_f1"],
                    gold_acc=result["category_metrics"]["gold_board"]["accuracy"],
                    gold_macro_f1=result["category_metrics"]["gold_board"]["macro_f1"],
                    no_skill_f1=result["category_metrics"]["no_skill_needed"]["f1"],
                )
            )

        if full_results:
            lines.extend(
                [
                    "",
                    "### Prototype Drop Relative To Full Bank",
                    "",
                ]
            )
            for backend_name in sorted(set(full_results) & set(prototype_results)):
                full_macro_f1 = full_results[backend_name]["category_metrics"]["end_to_end"]["macro_f1"]
                prototype_macro_f1 = prototype_results[backend_name]["category_metrics"]["end_to_end"][
                    "macro_f1"
                ]
                lines.append(
                    f"- {backend_name}: full={full_macro_f1:.4f}, prototype={prototype_macro_f1:.4f}, drop={prototype_macro_f1 - full_macro_f1:.4f}"
                )

    with open(os.path.join(output_dir, "error_analysis.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def build_output_dir(raw_output_dir: str, smoke: bool) -> str:
    normalized = raw_output_dir.replace("/", os.sep)
    absolute_path = normalized
    if not os.path.isabs(absolute_path):
        absolute_path = os.path.join(path_store.root, normalized)
    if smoke:
        absolute_path = os.path.join(absolute_path, "smoke")
    return absolute_path


def discover_base_task_ids(source_experiment_name: str) -> list[str]:
    source_root = os.path.join(path_store.experiment_outputs, source_experiment_name, "tasks")
    return sorted(
        {
            get_base_task_id(task_name)
            for task_name in os.listdir(source_root)
            if os.path.isdir(os.path.join(source_root, task_name))
        }
    )


def main() -> None:
    args = parse_args()
    update_root(args.root)
    ensure_package_installed("appworld_experiments")

    output_dir = build_output_dir(args.output_dir, smoke=args.smoke)
    os.makedirs(output_dir, exist_ok=True)

    selected_base_task_ids = None
    if args.smoke:
        selected_base_task_ids = discover_base_task_ids(args.source_experiment)[:8]

    datapoints, reconstruction_stats = load_or_reconstruct_datapoints(
        source_experiment_name=args.source_experiment,
        output_dir=output_dir,
        allowed_base_task_ids=set(selected_base_task_ids) if selected_base_task_ids is not None else None,
    )

    if args.smoke:
        write_json(
            {
                "selected_base_task_ids": selected_base_task_ids,
                "generated_at_utc": current_utc_timestamp(),
            },
            os.path.join(output_dir, "smoke_manifest.json"),
            silent=True,
        )

    split_manifest = build_split_manifest(
        datapoints=datapoints,
        output_dir=output_dir,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
    )
    train_base_task_ids = set(split_manifest["train_base_task_ids"])
    validation_base_task_ids = set(split_manifest["validation_base_task_ids"])
    train_datapoints = [item for item in datapoints if item["base_task_id"] in train_base_task_ids]
    validation_datapoints = [
        item for item in datapoints if item["base_task_id"] in validation_base_task_ids
    ]
    write_bank_stats(output_dir, train_datapoints, validation_datapoints)

    if args.stage in {"full_bank", "all"}:
        run_full_bank(
            output_dir=output_dir,
            train_datapoints=train_datapoints,
            validation_datapoints=validation_datapoints,
            backends=args.backends,
            top_k=args.top_k,
        )

    if args.stage in {"prototype_bank", "all"}:
        run_prototype_bank(
            output_dir=output_dir,
            train_datapoints=train_datapoints,
            validation_datapoints=validation_datapoints,
            backends=args.prototype_backends,
            top_k=args.top_k,
            prototype_k_max=args.prototype_k_max,
        )

    # The old markdown summary is tied to the v1 taxonomy and will be rewritten separately.

    print(
        f"[text_similarity_retrieval] stage={args.stage} output_dir={output_dir} "
        f"train={len(train_datapoints)} val={len(validation_datapoints)}"
    )


if __name__ == "__main__":
    main()

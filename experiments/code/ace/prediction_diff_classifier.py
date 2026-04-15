import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any

from appworld import AppWorld
from appworld.common.path_store import path_store
from appworld.common.utils import (
    FromDict,
    chunk_and_return,
    read_file,
    write_json,
    write_jsonl,
    yield_jsonl,
)
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.playbook import extract_json_from_text


PREDICTION_TRIGGER_TEXT = "Before writing your next code block, predict what the environment would return"
ALLOWED_PRIMARY_BOARDS = {
    "docs_lookup",
    "auth",
    "read_fetch",
    "local_reasoning",
    "write_or_complete",
    "other",
}
ALLOWED_DIFF_CATEGORIES = {
    "match",
    "safe_abstraction",
    "schema_or_name_mismatch",
    "missing_decisive_information",
    "value_or_state_mismatch",
    "wrong_action_or_failure_mode",
    "other",
}


class PredictionDiffClassifier(FromDict):
    def __init__(
        self,
        classifier_model_config: dict,
        classifier_prompt_file_path: str,
        source_experiment_name: str,
        max_interactions_per_task: int | None = None,
        max_field_chars: int = 6000,
        max_history_chars: int = 35000,
        log_lm_calls: bool = True,
        prediction_trigger_text: str = PREDICTION_TRIGGER_TEXT,
    ):
        self.classifier_model = LiteLLMGenerator(**classifier_model_config)
        self.classifier_prompt_file_path = classifier_prompt_file_path
        self.classifier_prompt_template = read_file(
            classifier_prompt_file_path.replace("/", os.sep)
        )
        self.source_experiment_name = source_experiment_name
        self.max_interactions_per_task = max_interactions_per_task
        self.max_field_chars = max_field_chars
        self.max_history_chars = max_history_chars
        self.log_lm_calls = log_lm_calls
        self.prediction_trigger_text = prediction_trigger_text

    def solve_tasks(
        self,
        task_ids: list[str],
        experiment_name: str | None = None,
        num_processes: int = 1,
        process_index: int | None = 0,
    ) -> None:
        if not experiment_name:
            raise ValueError("experiment_name must be provided for prediction diff classification.")

        process_index = process_index or 0
        num_processes = num_processes or 1
        num_tasks = len(task_ids)
        num_processes = min(num_processes, num_tasks) if num_tasks else 1
        chunked_task_ids = (
            chunk_and_return(task_ids, num_chunks=num_processes, chunk_index=process_index)
            if task_ids
            else []
        )

        print(
            f"[prediction_diff] source_experiment={self.source_experiment_name} "
            f"tasks={len(chunked_task_ids)} "
            f"(process {process_index + 1}/{num_processes})"
        )
        task_summaries: list[dict[str, Any]] = []

        for task_id in chunked_task_ids:
            summary = self.classify_task(task_id=task_id)
            if summary is not None:
                task_summaries.append(summary)

        self.write_aggregate_outputs(
            task_summaries=task_summaries,
            task_ids=chunked_task_ids,
            process_index=process_index,
            num_processes=num_processes,
        )

    def classify_task(self, task_id: str) -> dict[str, Any] | None:
        source_logs_dir = self._source_logs_dir(task_id)
        environment_log_path = os.path.join(source_logs_dir, "environment_io.md")
        predicted_log_path = os.path.join(source_logs_dir, "predicted_environment_io.md")
        lm_calls_path = os.path.join(source_logs_dir, "lm_calls.jsonl")
        evaluation_report_path = self._source_evaluation_report_path(task_id)

        required_paths = [environment_log_path, predicted_log_path, lm_calls_path]
        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        if missing_paths:
            print(f"[prediction_diff] skip task={task_id}, missing files: {missing_paths}")
            return None

        environment_entries = AppWorld.parse_environment_io_log(file_path=environment_log_path)
        total_interaction_count = len(environment_entries)
        predicted_entries = AppWorld.parse_environment_io_log(file_path=predicted_log_path)
        lm_calls = list(yield_jsonl(lm_calls_path))
        reconstructed_steps = self.extract_reconstructed_steps(lm_calls)
        evaluation_summary = self.parse_evaluation_report(evaluation_report_path)

        if self.max_interactions_per_task is not None:
            environment_entries = environment_entries[: self.max_interactions_per_task]
            predicted_entries = predicted_entries[: self.max_interactions_per_task]

        interactions = self.build_aligned_interactions(
            task_id=task_id,
            environment_entries=environment_entries,
            predicted_entries=predicted_entries,
            reconstructed_steps=reconstructed_steps,
        )

        task_output_logs_dir = self._target_task_logs_dir(task_id)
        os.makedirs(task_output_logs_dir, exist_ok=True)
        if self.log_lm_calls:
            classifier_lm_calls_path = os.path.join(task_output_logs_dir, "classifier_lm_calls.jsonl")
            self.classifier_model.log_calls_to(file_path=classifier_lm_calls_path)

        interaction_records: list[dict[str, Any]] = []
        for interaction in interactions:
            output_record = self.classify_interaction(
                task_id=task_id,
                interaction=interaction,
            )
            interaction_records.append(output_record)

        file_suffix = self.build_partial_suffix(
            processed_interaction_count=len(interaction_records),
            total_interaction_count=total_interaction_count,
        )
        output_path = os.path.join(
            task_output_logs_dir,
            f"prediction_diff_classification{file_suffix}.jsonl",
        )
        write_jsonl(interaction_records, output_path, silent=True)
        print(
            f"[prediction_diff] task={task_id} interactions={len(interaction_records)} "
            f"output={output_path}"
        )
        return self.build_task_summary(
            task_id,
            interaction_records,
            evaluation_summary,
            file_suffix=file_suffix,
        )

    def classify_interaction(
        self,
        task_id: str,
        interaction: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self.render_classifier_prompt(task_id=task_id, interaction=interaction)
        model_output = self.classifier_model.generate(messages=[{"role": "user", "content": prompt}])
        raw_response = model_output.get("content") or ""
        parsed_response = extract_json_from_text(raw_response) or {}

        normalized = self.normalize_classification_response(
            parsed_response=parsed_response,
            interaction=interaction,
            raw_response=raw_response,
        )
        output_record = {
            "interaction_index": interaction["interaction_index"],
            "primary_board": normalized["primary_board"],
            "diff_category": normalized["diff_category"],
            "reasoning": normalized["reasoning"],
            "evidence": normalized["evidence"],
            "model_response_raw": normalized["model_response_raw"],
            "parse_errors": normalized["parse_errors"],
        }
        return output_record

    def render_classifier_prompt(self, task_id: str, interaction: dict[str, Any]) -> str:
        prior_trajectory = self.format_prior_trajectory(interaction["history"])
        replacements = {
            "{{task_id}}": task_id,
            "{{source_experiment_name}}": self.source_experiment_name,
            "{{interaction_index}}": str(interaction["interaction_index"]),
            "{{prior_trajectory}}": prior_trajectory,
            "{{current_reasoning_text}}": self.clip_text(interaction["current_reasoning"]),
            "{{current_code}}": interaction["current_code"],
            "{{predicted_output}}": self.clip_text(interaction["predicted_output"]),
            "{{actual_output}}": self.clip_text(interaction["actual_output"]),
        }
        prompt = self.classifier_prompt_template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        return prompt

    def format_prior_trajectory(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return "(no prior interactions)"

        chunks: list[str] = []
        for item in history:
            code = item.get("code") or ""
            output = self.clip_text(item.get("actual_output") or "")
            chunk = "\n".join(
                [
                    f"[Interaction {item['interaction_index']}]",
                    "Code:",
                    "```python",
                    code,
                    "```",
                    "Output:",
                    "```",
                    output,
                    "```",
                ]
            )
            chunks.append(chunk)
        joined = "\n\n".join(chunks)
        return self.clip_text(joined, max_chars=self.max_history_chars)

    def normalize_classification_response(
        self,
        parsed_response: dict[str, Any],
        interaction: dict[str, Any],
        raw_response: str,
    ) -> dict[str, Any]:
        primary_board = str(parsed_response.get("primary_board", "")).strip().lower()
        diff_category = str(parsed_response.get("diff_category", "")).strip().lower()
        reasoning = str(parsed_response.get("reasoning", "")).strip()
        evidence = parsed_response.get("evidence") if isinstance(parsed_response.get("evidence"), dict) else {}

        parse_errors: list[str] = []
        if primary_board not in ALLOWED_PRIMARY_BOARDS:
            parse_errors.append(f"invalid primary_board={primary_board}")
            primary_board = self.infer_primary_board(interaction["current_code"])
        if diff_category not in ALLOWED_DIFF_CATEGORIES:
            parse_errors.append(f"invalid diff_category={diff_category}")
            diff_category = self.infer_diff_category(interaction)
        if not reasoning:
            reasoning = "Classifier returned empty reasoning; fallback reasoning was applied."
            parse_errors.append("empty reasoning")
        if not evidence:
            evidence = {
                "current_reasoning_summary": self.clip_text(interaction["current_reasoning"], max_chars=400),
                "key_predicted_claim": self.clip_text(interaction["predicted_output"], max_chars=400),
                "key_actual_fact": self.clip_text(interaction["actual_output"], max_chars=400),
            }
            parse_errors.append("empty or invalid evidence")

        normalized = {
            "primary_board": primary_board,
            "diff_category": diff_category,
            "reasoning": reasoning,
            "evidence": evidence,
            "model_response_raw": raw_response,
            "parse_errors": parse_errors,
        }
        return normalized

    def infer_primary_board(self, code: str) -> str:
        lowered = (code or "").lower()
        if "api_docs.show_api_doc" in lowered or "api_docs.show_api_descriptions" in lowered:
            return "docs_lookup"
        if ".login(" in lowered or "show_account_passwords" in lowered:
            return "auth"
        if "complete_task(" in lowered:
            return "write_or_complete"
        write_markers = [
            ".create_",
            ".update_",
            ".delete_",
            ".remove_",
            ".move_",
            ".send_",
            ".like_",
            ".follow_",
            ".review_",
            ".approve_",
            ".deny_",
        ]
        if any(marker in lowered for marker in write_markers):
            return "write_or_complete"
        read_markers = [".show_", ".search_", ".download_", ".get_"]
        if any(marker in lowered for marker in read_markers):
            return "read_fetch"
        local_reasoning_markers = [
            "max(",
            "min(",
            "sorted(",
            "len(",
            "set(",
            "sum(",
            "append(",
            "extend(",
            "for ",
            "if ",
            "print(",
        ]
        if any(marker in lowered for marker in local_reasoning_markers):
            return "local_reasoning"
        return "other"

    def infer_diff_category(self, interaction: dict[str, Any]) -> str:
        predicted = (interaction["predicted_output"] or "").strip()
        actual = (interaction["actual_output"] or "").strip()
        board = self.infer_primary_board(interaction["current_code"])
        if self.normalize_text(predicted) == self.normalize_text(actual):
            return "match"
        if not predicted:
            return "missing_decisive_information"
        if (
            board != "docs_lookup"
            and self.looks_like_error(predicted) != self.looks_like_error(actual)
        ):
            return "wrong_action_or_failure_mode"
        if board in {"auth", "write_or_complete"} and self.looks_like_safe_success_abstraction(
            predicted, actual
        ):
            return "safe_abstraction"
        if self.looks_like_schema_mismatch(predicted, actual):
            return "schema_or_name_mismatch"
        if board != "docs_lookup" and self.misses_decisive_information(predicted, actual):
            return "missing_decisive_information"
        if "message" in actual.lower() and "message" in predicted.lower():
            return "safe_abstraction"
        if self.looks_like_value_or_state_mismatch(predicted, actual):
            return "value_or_state_mismatch"
        return "other"

    def looks_like_schema_mismatch(self, predicted: str, actual: str) -> bool:
        predicted_keys = set(re.findall(r'"([a-zA-Z0-9_]+)"\s*:', predicted))
        actual_keys = set(re.findall(r'"([a-zA-Z0-9_]+)"\s*:', actual))
        if not predicted_keys or not actual_keys:
            return False
        intersection = predicted_keys & actual_keys
        union = predicted_keys | actual_keys
        jaccard = len(intersection) / max(1, len(union))
        return jaccard < 0.4

    def looks_like_value_or_state_mismatch(self, predicted: str, actual: str) -> bool:
        predicted_lower = (predicted or "").lower()
        actual_lower = (actual or "").lower()

        predicted_keys = set(re.findall(r'"([a-zA-Z0-9_]+)"\s*:', predicted))
        actual_keys = set(re.findall(r'"([a-zA-Z0-9_]+)"\s*:', actual))
        shared_keys = predicted_keys & actual_keys
        if shared_keys:
            return True

        if predicted.startswith("[") and actual.startswith("["):
            return True
        if predicted.startswith("{") and actual.startswith("{"):
            return True

        content_markers = [
            "access_token",
            "password",
            "title",
            "name",
            "email",
            "message",
            "count",
            "status",
            "created_at",
        ]
        shared_markers = [
            marker for marker in content_markers
            if marker in predicted_lower and marker in actual_lower
        ]
        return len(shared_markers) >= 2

    def looks_like_error(self, text: str) -> bool:
        lowered = (text or "").lower()
        error_markers = [
            "traceback",
            "exception",
            "error",
            "failed",
            "failure",
            "invalid",
            "unauthorized",
            "forbidden",
            "denied",
            "not found",
            "does not exist",
            "missing required",
            "already exists",
        ]
        return any(marker in lowered for marker in error_markers)

    def looks_like_safe_success_abstraction(self, predicted: str, actual: str) -> bool:
        predicted_lower = (predicted or "").lower()
        actual_lower = (actual or "").lower()
        if self.looks_like_error(predicted) or self.looks_like_error(actual):
            return False
        if len(predicted) >= len(actual):
            return False
        success_signals = [
            "execution successful",
            "access_token",
            "\"message\"",
            "marked the active task complete",
        ]
        return any(signal in predicted_lower and signal in actual_lower for signal in success_signals)

    def misses_decisive_information(self, predicted: str, actual: str) -> bool:
        predicted_lower = (predicted or "").lower()
        actual_lower = (actual or "").lower()
        decisive_markers = [
            "access_token",
            "_id",
            "\"id\"",
            "created_at",
            "updated_at",
            "\"status\"",
            "sender",
            "receiver",
            "email",
            "phone_number",
            "destination_file_path",
        ]
        missing_markers = [
            marker for marker in decisive_markers
            if marker in actual_lower and marker not in predicted_lower
        ]
        if not missing_markers:
            return False
        high_signal_markers = {
            "access_token",
            "_id",
            "\"id\"",
            "\"status\"",
            "destination_file_path",
        }
        if any(marker in high_signal_markers for marker in missing_markers):
            return True
        return len(missing_markers) >= 2

    def build_task_summary(
        self,
        task_id: str,
        interaction_records: list[dict[str, Any]],
        evaluation_summary: dict[str, Any],
        file_suffix: str = "",
    ) -> dict[str, Any]:
        board_counter = Counter(record["primary_board"] for record in interaction_records)
        diff_counter = Counter(record["diff_category"] for record in interaction_records)
        non_match_count = sum(
            1 for record in interaction_records if record["diff_category"] != "match"
        )
        interaction_count = len(interaction_records)
        return {
            "task_id": task_id,
            "task_passed": evaluation_summary.get("task_passed"),
            "num_passed_tests": evaluation_summary.get("num_passed_tests"),
            "num_failed_tests": evaluation_summary.get("num_failed_tests"),
            "output_file_suffix": file_suffix,
            "interaction_count": interaction_count,
            "non_match_count": non_match_count,
            "non_match_ratio": (
                non_match_count / interaction_count if interaction_count > 0 else None
            ),
            "primary_board_counts": dict(board_counter),
            "diff_category_counts": dict(diff_counter),
        }

    def write_aggregate_outputs(
        self,
        task_summaries: list[dict[str, Any]],
        task_ids: list[str],
        process_index: int,
        num_processes: int,
    ) -> None:
        aggregate_suffix = self.build_aggregate_suffix(task_summaries)
        analysis_dir = os.path.join(
            path_store.experiment_outputs, self.source_experiment_name, "analysis"
        )
        os.makedirs(analysis_dir, exist_ok=True)

        task_summary_path = os.path.join(
            analysis_dir,
            f"task_level_prediction_diff_summary{aggregate_suffix}.jsonl",
        )
        write_jsonl(task_summaries, task_summary_path, silent=True)

        stats = self.compute_stats(task_summaries)
        stats["metadata"] = {
            "generated_at_utc": datetime.utcnow().isoformat(),
            "source_experiment_name": self.source_experiment_name,
            "processed_task_count": len(task_summaries),
            "requested_task_count": len(task_ids),
            "process_index": process_index,
            "num_processes": num_processes,
        }
        stats_path = os.path.join(
            analysis_dir,
            f"prediction_diff_stats{aggregate_suffix}.json",
        )
        write_json(stats, stats_path, silent=True)

        markdown_report = self.render_markdown_report(stats)
        markdown_path = os.path.join(
            analysis_dir,
            f"prediction_diff_stats{aggregate_suffix}.md",
        )
        with open(markdown_path, "w", encoding="utf-8") as file:
            file.write(markdown_report)

        print(
            f"[prediction_diff] wrote aggregate outputs: "
            f"{task_summary_path}, {stats_path}, {markdown_path}"
        )

    def compute_stats(self, task_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        if not task_summaries:
            return {
                "task_count": 0,
                "passed_task_count": 0,
                "failed_task_count": 0,
                "feature_stats": [],
            }

        failed_task_ids = {
            item["task_id"] for item in task_summaries if item.get("task_passed") is False
        }
        passed_task_ids = {
            item["task_id"] for item in task_summaries if item.get("task_passed") is True
        }

        feature_names = self.collect_feature_names(task_summaries)
        feature_stats = []
        for feature_name in feature_names:
            counts = [self.get_feature_count(item, feature_name) for item in task_summaries]
            failures = [1 if item.get("task_passed") is False else 0 for item in task_summaries]
            feature_stats.append(
                self.compute_feature_stat_row(
                    feature_name=feature_name,
                    task_summaries=task_summaries,
                    counts=counts,
                    failures=failures,
                )
            )

        feature_stats.sort(
            key=lambda item: (
                item.get("odds_ratio") if item.get("odds_ratio") is not None else -1.0
            ),
            reverse=True,
        )

        return {
            "task_count": len(task_summaries),
            "passed_task_count": len(passed_task_ids),
            "failed_task_count": len(failed_task_ids),
            "feature_stats": feature_stats,
        }

    def collect_feature_names(self, task_summaries: list[dict[str, Any]]) -> list[str]:
        names = set()
        for summary in task_summaries:
            names.add("non_match_count")
            for key in summary.get("primary_board_counts", {}).keys():
                names.add(f"primary_board::{key}")
            for key in summary.get("diff_category_counts", {}).keys():
                names.add(f"diff_category::{key}")
        return sorted(names)

    def get_feature_count(self, task_summary: dict[str, Any], feature_name: str) -> int:
        if feature_name == "non_match_count":
            return int(task_summary.get("non_match_count") or 0)
        if feature_name.startswith("primary_board::"):
            key = feature_name.split("::", 1)[1]
            return int(task_summary.get("primary_board_counts", {}).get(key, 0))
        if feature_name.startswith("diff_category::"):
            key = feature_name.split("::", 1)[1]
            return int(task_summary.get("diff_category_counts", {}).get(key, 0))
        return 0

    def compute_feature_stat_row(
        self,
        feature_name: str,
        task_summaries: list[dict[str, Any]],
        counts: list[int],
        failures: list[int],
    ) -> dict[str, Any]:
        present_mask = [count > 0 for count in counts]
        a = sum(1 for present, failed in zip(present_mask, failures) if present and failed == 1)
        b = sum(1 for present, failed in zip(present_mask, failures) if present and failed == 0)
        c = sum(1 for present, failed in zip(present_mask, failures) if (not present) and failed == 1)
        d = sum(1 for present, failed in zip(present_mask, failures) if (not present) and failed == 0)

        fail_rate_with = a / (a + b) if (a + b) > 0 else None
        fail_rate_without = c / (c + d) if (c + d) > 0 else None
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        failed_counts = [count for count, failed in zip(counts, failures) if failed == 1]
        passed_counts = [count for count, failed in zip(counts, failures) if failed == 0]
        mean_count_failed = (
            sum(failed_counts) / len(failed_counts) if len(failed_counts) > 0 else None
        )
        mean_count_passed = (
            sum(passed_counts) / len(passed_counts) if len(passed_counts) > 0 else None
        )
        spearman = self.spearman_correlation(counts, failures)

        return {
            "feature_name": feature_name,
            "tasks_with_feature": a + b,
            "tasks_without_feature": c + d,
            "failed_with_feature": a,
            "passed_with_feature": b,
            "failed_without_feature": c,
            "passed_without_feature": d,
            "failure_rate_with_feature": fail_rate_with,
            "failure_rate_without_feature": fail_rate_without,
            "failure_rate_lift": (
                fail_rate_with - fail_rate_without
                if fail_rate_with is not None and fail_rate_without is not None
                else None
            ),
            "odds_ratio": odds_ratio,
            "mean_count_failed_tasks": mean_count_failed,
            "mean_count_passed_tasks": mean_count_passed,
            "spearman_count_vs_failure": spearman,
        }

    def spearman_correlation(self, x_values: list[int], y_values: list[int]) -> float | None:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return None
        x_ranks = self.rank_with_ties(x_values)
        y_ranks = self.rank_with_ties(y_values)
        return self.pearson_correlation(x_ranks, y_ranks)

    def rank_with_ties(self, values: list[int]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(indexed):
            tie_start = index
            tie_value = indexed[index][1]
            while index < len(indexed) and indexed[index][1] == tie_value:
                index += 1
            tie_end = index - 1
            average_rank = (tie_start + tie_end) / 2.0 + 1.0
            for tie_index in range(tie_start, tie_end + 1):
                original_index = indexed[tie_index][0]
                ranks[original_index] = average_rank
        return ranks

    def pearson_correlation(self, x_values: list[float], y_values: list[float]) -> float | None:
        n = len(x_values)
        if n < 2:
            return None
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
        y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
        if x_denom == 0.0 or y_denom == 0.0:
            return None
        return numerator / (x_denom * y_denom)

    def render_markdown_report(self, stats: dict[str, Any]) -> str:
        lines = [
            "# Prediction Diff Statistical Analysis",
            "",
            f"- task_count: {stats.get('task_count', 0)}",
            f"- passed_task_count: {stats.get('passed_task_count', 0)}",
            f"- failed_task_count: {stats.get('failed_task_count', 0)}",
            "",
            "## Top Failure-Associated Features",
            "",
            "| feature | fail_rate_with | fail_rate_without | lift | odds_ratio | spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        feature_stats = stats.get("feature_stats", [])
        for row in feature_stats[:30]:
            lines.append(
                "| {feature} | {with_rate} | {without_rate} | {lift} | {or_} | {rho} |".format(
                    feature=row.get("feature_name"),
                    with_rate=self.format_number(row.get("failure_rate_with_feature")),
                    without_rate=self.format_number(row.get("failure_rate_without_feature")),
                    lift=self.format_number(row.get("failure_rate_lift")),
                    or_=self.format_number(row.get("odds_ratio")),
                    rho=self.format_number(row.get("spearman_count_vs_failure")),
                )
            )
        lines.append("")
        return "\n".join(lines)

    def format_number(self, value: float | None) -> str:
        if value is None:
            return "NA"
        return f"{value:.4f}"

    def build_aligned_interactions(
        self,
        task_id: str,
        environment_entries: list[dict[str, str]],
        predicted_entries: list[dict[str, str]],
        reconstructed_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        interactions: list[dict[str, Any]] = []
        step_cursor = 0

        for index, environment_entry in enumerate(environment_entries):
            current_code = environment_entry.get("input") or ""
            current_actual_output = environment_entry.get("output") or ""
            current_predicted_clipped = (
                predicted_entries[index]["output"] if index < len(predicted_entries) else ""
            )

            matched_step, step_cursor, alignment = self.match_step_for_code(
                current_code=current_code,
                reconstructed_steps=reconstructed_steps,
                step_cursor=step_cursor,
            )

            current_reasoning = matched_step.get("reasoning") if matched_step else ""
            predicted_output_raw = matched_step.get("prediction_raw") if matched_step else ""
            prediction_source = "raw_from_lm_calls" if predicted_output_raw else "clipped_fallback"
            if not predicted_output_raw:
                predicted_output_raw = current_predicted_clipped

            history = []
            for past_index in range(index):
                past_entry = environment_entries[past_index]
                history.append(
                    {
                        "interaction_index": past_index + 1,
                        "code": past_entry.get("input") or "",
                        "actual_output": past_entry.get("output") or "",
                    }
                )

            interactions.append(
                {
                    "task_id": task_id,
                    "interaction_index": index + 1,
                    "current_reasoning": current_reasoning,
                    "current_code": current_code,
                    "predicted_output": predicted_output_raw,
                    "actual_output": current_actual_output,
                    "history": history,
                    "prediction_source": prediction_source,
                    "alignment": alignment,
                }
            )
        return interactions

    def match_step_for_code(
        self,
        current_code: str,
        reconstructed_steps: list[dict[str, Any]],
        step_cursor: int,
        lookahead: int = 4,
    ) -> tuple[dict[str, Any] | None, int, dict[str, Any]]:
        normalized_current = self.normalize_code(current_code)
        best_match_index = None
        upper = min(len(reconstructed_steps), step_cursor + lookahead)

        for candidate_index in range(step_cursor, upper):
            candidate_code = self.normalize_code(reconstructed_steps[candidate_index].get("code") or "")
            if candidate_code == normalized_current:
                best_match_index = candidate_index
                break

        if best_match_index is None and step_cursor < len(reconstructed_steps):
            best_match_index = step_cursor
            matched = reconstructed_steps[best_match_index]
            return (
                matched,
                best_match_index + 1,
                {
                    "matched": False,
                    "strategy": "cursor_fallback",
                    "step_index": best_match_index,
                    "step_code_match": self.normalize_code(matched.get("code") or "")
                    == normalized_current,
                },
            )

        if best_match_index is None:
            return None, step_cursor, {"matched": False, "strategy": "no_step_available"}

        matched = reconstructed_steps[best_match_index]
        return (
            matched,
            best_match_index + 1,
            {"matched": True, "strategy": "exact_in_lookahead", "step_index": best_match_index},
        )

    def extract_reconstructed_steps(self, lm_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        steps = []
        total_calls = len(lm_calls)
        call_index = 0
        while call_index < total_calls:
            lm_call = lm_calls[call_index]
            if self.is_prediction_call(lm_call):
                call_index += 1
                continue

            output_content = self.extract_output_content(lm_call)
            code, reasoning = self.extract_code_and_reasoning(output_content)
            if not code.strip():
                call_index += 1
                continue

            step = {
                "lm_call_index": call_index,
                "code": code,
                "reasoning": reasoning,
                "prediction_raw": "",
                "prediction_lm_call_index": None,
            }

            next_index = call_index + 1
            while next_index < total_calls:
                next_call = lm_calls[next_index]
                if self.is_prediction_call(next_call):
                    step["prediction_raw"] = self.clean_predicted_output(
                        self.extract_output_content(next_call)
                    )
                    step["prediction_lm_call_index"] = next_index
                    break
                next_output_content = self.extract_output_content(next_call)
                next_code, _ = self.extract_code_and_reasoning(next_output_content)
                if next_code.strip():
                    break
                next_index += 1

            steps.append(step)
            call_index += 1
        return steps

    def is_prediction_call(self, lm_call: dict[str, Any]) -> bool:
        messages = lm_call["input"]["messages"]
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            if self.prediction_trigger_text in content:
                return True
        return False

    def extract_output_content(self, lm_call: dict[str, Any]) -> str:
        message = lm_call.get("output", {}).get("choices", [{}])[0].get("message", {})
        return str(message.get("content") or "")

    def extract_code_and_reasoning(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""

        full_match = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL)
        if full_match:
            code = full_match.group(1).strip()
            reasoning = text[: full_match.start()].strip()
            reasoning = re.sub(r"\n*Code:\s*$", "", reasoning).strip()
            return code, reasoning

        partial_match = re.search(r"```python\n(.*)$", text, flags=re.DOTALL)
        if partial_match:
            code = partial_match.group(1).strip()
            reasoning = text[: partial_match.start()].strip()
            reasoning = re.sub(r"\n*Code:\s*$", "", reasoning).strip()
            return code, reasoning

        return "", text.strip()

    def clean_predicted_output(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        verbose_prefix = "I think the environment output for this code execution is:"
        if cleaned.startswith(verbose_prefix):
            cleaned = cleaned[len(verbose_prefix) :].lstrip()

        if cleaned.startswith("Output:\n"):
            cleaned = cleaned[len("Output:\n") :].lstrip()
        elif cleaned.startswith("Output:"):
            cleaned = cleaned[len("Output:") :].lstrip()

        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.startswith("python\n"):
                cleaned = cleaned[len("python\n") :]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]

        return cleaned.rstrip()

    def parse_evaluation_report(self, report_path: str) -> dict[str, Any]:
        if not os.path.exists(report_path):
            return {"task_passed": None, "num_passed_tests": None, "num_failed_tests": None}

        report_text = read_file(report_path)
        passed_match = re.search(r"Num Passed Tests\s*:\s*(\d+)", report_text)
        failed_match = re.search(r"Num Failed Tests\s*:\s*(\d+)", report_text)
        total_match = re.search(r"Num Total\s+Tests\s*:\s*(\d+)", report_text)

        num_passed = int(passed_match.group(1)) if passed_match else None
        num_failed = int(failed_match.group(1)) if failed_match else None
        _ = int(total_match.group(1)) if total_match else None
        task_passed = None
        if num_failed is not None:
            task_passed = num_failed == 0
        return {
            "task_passed": task_passed,
            "num_passed_tests": num_passed,
            "num_failed_tests": num_failed,
        }

    def normalize_code(self, code: str) -> str:
        lines = [(line.rstrip()) for line in (code or "").strip().splitlines()]
        return "\n".join(lines).strip()

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def clip_text(self, text: str, max_chars: int | None = None) -> str:
        max_chars = max_chars if max_chars is not None else self.max_field_chars
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars]
        return clipped + "\n[TRUNCATED]"

    def _source_logs_dir(self, task_id: str) -> str:
        return os.path.join(
            path_store.experiment_outputs,
            self.source_experiment_name,
            "tasks",
            task_id,
            "logs",
        )

    def _source_evaluation_report_path(self, task_id: str) -> str:
        return os.path.join(
            path_store.experiment_outputs,
            self.source_experiment_name,
            "tasks",
            task_id,
            "evaluation",
            "report.md",
        )

    def _target_task_logs_dir(self, task_id: str) -> str:
        return os.path.join(
            path_store.experiment_outputs, self.source_experiment_name, "tasks", task_id, "logs"
        )

    def build_partial_suffix(
        self, processed_interaction_count: int, total_interaction_count: int
    ) -> str:
        if (
            self.max_interactions_per_task is not None
            and processed_interaction_count < total_interaction_count
        ):
            return f".partial_top_{processed_interaction_count}"
        return ""

    def build_aggregate_suffix(self, task_summaries: list[dict[str, Any]]) -> str:
        all_suffixes = {summary.get("output_file_suffix", "") for summary in task_summaries}
        non_empty_suffixes = {suffix for suffix in all_suffixes if suffix}
        if "" in all_suffixes and non_empty_suffixes:
            return ".partial_mixed"
        if len(non_empty_suffixes) == 1:
            return non_empty_suffixes.pop()
        if len(non_empty_suffixes) > 1:
            return ".partial_mixed"
        return ""

import json
import os
from typing import Any

from appworld import AppWorld
from appworld.common.path_store import path_store
from appworld.common.utils import FromDict, read_file, read_json, write_json, yield_jsonl
from appworld.task import Task
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.playbook import extract_json_from_text
from appworld_experiments.code.ace.prediction_diff_reconstruction import (
    PREDICTION_TRIGGER_TEXT,
    build_aligned_interactions,
    extract_reconstructed_steps,
)
from appworld_experiments.code.ace.skillbank import (
    DIFF_CATEGORIES,
    PRIMARY_BOARDS,
    apply_skill_operations,
    build_empty_skillbank,
    count_skills,
    ensure_skillbank_shape,
    get_bucket,
)


POST_SKILL_PRIMARY_BOARDS = {"auth", "local_reasoning"}
UNSUPPORTED_PRIMARY_BOARDS = {"other"}
SKIPPED_DIFF_CATEGORIES = {"", "match"}


class PredictionDiffCurator(FromDict):
    def __init__(
        self,
        curator_model_config: dict,
        curator_prompt_file_path: str,
        initial_skillbank_file_path: str,
        trained_skillbank_file_path: str,
        source_experiment_name: str,
        classification_file_name: str = "prediction_diff_classification.jsonl",
        max_field_chars: int = 10000,
        max_history_chars: int = 100000,
        log_lm_calls: bool = True,
        prediction_trigger_text: str = PREDICTION_TRIGGER_TEXT,
    ):
        self.curator_model_config = curator_model_config
        self.curator_model: LiteLLMGenerator | None = None
        self.curator_prompt_file_path = curator_prompt_file_path
        self.curator_prompt_template = read_file(curator_prompt_file_path.replace("/", os.sep))
        self.initial_skillbank_file_path = initial_skillbank_file_path
        self.trained_skillbank_file_path = trained_skillbank_file_path
        self.source_experiment_name = source_experiment_name
        self.classification_file_name = classification_file_name
        self.max_field_chars = max_field_chars
        self.max_history_chars = max_history_chars
        self.log_lm_calls = log_lm_calls
        self.prediction_trigger_text = prediction_trigger_text
        self.skillbank = build_empty_skillbank()
        self.current_task_index = 0

    def _get_curator_model(self) -> LiteLLMGenerator:
        if self.curator_model is None:
            self.curator_model = LiteLLMGenerator(**self.curator_model_config)
        return self.curator_model

    def solve_tasks(
        self,
        task_ids: list[str],
        experiment_name: str | None = None,
        num_processes: int = 1,
        process_index: int | None = 0,
    ) -> None:
        if num_processes != 1 or (process_index or 0) != 0:
            raise ValueError(
                "prediction-diff-curation must run in a single sequential process because "
                "skillbank updates are sequential across tasks."
            )

        self.initialize_skillbank()
        print(
            f"[prediction_diff_curation] source_experiment={self.source_experiment_name} "
            f"tasks={len(task_ids)} skillbank={self.trained_skillbank_file_path}"
        )
        for task_index, task_id in enumerate(task_ids):
            self.current_task_index = task_index
            self.curate_task(task_id)
            if (self.current_task_index + 1) % 30 == 0:
                self.save_skillbank_snapshot()

    def initialize_skillbank(self) -> None:
        if os.path.exists(self.initial_skillbank_file_path):
            self.skillbank = ensure_skillbank_shape(read_json(self.initial_skillbank_file_path))
        else:
            self.skillbank = build_empty_skillbank()
        self.persist_skillbank()

    def curate_task(self, task_id: str) -> dict[str, Any] | None:
        source_logs_dir = self._source_logs_dir(task_id)
        environment_log_path = os.path.join(source_logs_dir, "environment_io.md")
        predicted_log_path = os.path.join(source_logs_dir, "predicted_environment_io.md")
        lm_calls_path = os.path.join(source_logs_dir, "lm_calls.jsonl")
        classification_path = os.path.join(source_logs_dir, self.classification_file_name)

        required_paths = [environment_log_path, predicted_log_path, lm_calls_path, classification_path]
        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        if missing_paths:
            print(f"[prediction_diff_curation] skip task={task_id}, missing files: {missing_paths}")
            return None

        classification_records = sorted(
            list(yield_jsonl(classification_path)),
            key=lambda item: int(item.get("interaction_index") or 0),
        )
        if not classification_records:
            print(
                f"[prediction_diff_curation] skip task={task_id}, empty classification file: "
                f"{classification_path}"
            )
            return None

        environment_entries = AppWorld.parse_environment_io_log(file_path=environment_log_path)
        predicted_entries = AppWorld.parse_environment_io_log(file_path=predicted_log_path)
        lm_calls = list(yield_jsonl(lm_calls_path))
        reconstructed_steps = extract_reconstructed_steps(
            lm_calls,
            prediction_trigger_text=self.prediction_trigger_text,
        )
        interactions = build_aligned_interactions(
            task_id=task_id,
            environment_entries=environment_entries,
            predicted_entries=predicted_entries,
            reconstructed_steps=reconstructed_steps,
        )
        interaction_lookup = {
            int(interaction["interaction_index"]): interaction for interaction in interactions
        }
        classification_lookup = {
            int(record["interaction_index"]): record for record in classification_records
        }

        task = Task.load(task_id=task_id)
        task_context = getattr(task, "instruction", "")
        task_output_logs_dir = self._target_task_logs_dir(task_id)
        os.makedirs(task_output_logs_dir, exist_ok=True)
        if self.log_lm_calls:
            curator_lm_calls_path = os.path.join(task_output_logs_dir, "curator_lm_calls.jsonl")
            self._get_curator_model().log_calls_to(file_path=curator_lm_calls_path)

        skill_count_before = count_skills(self.skillbank)
        interaction_outputs: list[dict[str, Any]] = []

        for classification_record in classification_records:
            interaction_index = int(classification_record.get("interaction_index") or 0)
            primary_board = str(classification_record.get("primary_board") or "").strip()
            diff_category = str(classification_record.get("diff_category") or "").strip()

            if primary_board not in PRIMARY_BOARDS:
                skip_reason = (
                    "non_curatable_primary_board"
                    if primary_board in UNSUPPORTED_PRIMARY_BOARDS
                    else "unsupported_primary_board"
                )
                interaction_outputs.append(
                    {
                        "interaction_index": interaction_index,
                        "primary_board": primary_board,
                        "diff_category": diff_category,
                        "skipped": True,
                        "skip_reason": skip_reason,
                    }
                )
                continue
            if diff_category in SKIPPED_DIFF_CATEGORIES:
                interaction_outputs.append(
                    {
                        "interaction_index": interaction_index,
                        "primary_board": primary_board,
                        "diff_category": diff_category,
                        "skipped": True,
                        "skip_reason": "skipped_diff_category",
                    }
                )
                continue
            if diff_category not in DIFF_CATEGORIES:
                interaction_outputs.append(
                    {
                        "interaction_index": interaction_index,
                        "primary_board": primary_board,
                        "diff_category": diff_category,
                        "skipped": True,
                        "skip_reason": "unsupported_diff_category",
                    }
                )
                continue

            current_interaction = interaction_lookup.get(interaction_index)
            if current_interaction is None:
                interaction_outputs.append(
                    {
                        "interaction_index": interaction_index,
                        "primary_board": primary_board,
                        "diff_category": diff_category,
                        "skipped": True,
                        "skip_reason": "missing_reconstructed_interaction",
                    }
                )
                continue

            next_interaction = None
            next_classification = None
            if primary_board in POST_SKILL_PRIMARY_BOARDS:
                next_interaction = interaction_lookup.get(interaction_index + 1)
                next_classification = classification_lookup.get(interaction_index + 1)

            current_bucket = get_bucket(self.skillbank, primary_board, diff_category)
            prompt = self.render_curator_prompt(
                task_context=task_context,
                previous_trajectory=self.format_previous_trajectory(current_interaction.get("history", [])),
                current_interaction=self.format_current_interaction(current_interaction),
                current_classification=self.format_current_classification(classification_record),
                current_skill_bucket=self.format_current_skill_bucket(
                    primary_board,
                    diff_category,
                    current_bucket,
                ),
                next_interaction=self.format_next_interaction(
                    next_interaction,
                    next_classification,
                ),
            )

            model_output = self._get_curator_model().generate(
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = model_output.get("content") or ""
            normalized = self.normalize_curator_response(
                raw_response=raw_response,
                current_bucket=current_bucket,
            )
            applied_operations = apply_skill_operations(
                self.skillbank,
                normalized["operations"],
                primary_board=primary_board,
                diff_category=diff_category,
                task_id=task_id,
            )

            interaction_outputs.append(
                {
                    "interaction_index": interaction_index,
                    "primary_board": primary_board,
                    "diff_category": diff_category,
                    "reasoning": normalized["reasoning"],
                    "operations": normalized["operations"],
                    "applied_operations": applied_operations,
                    "applied_operation_count": len(applied_operations),
                    "parse_errors": normalized["parse_errors"],
                    "model_response_raw": raw_response,
                    "bucket_skill_count_after": len(
                        get_bucket(self.skillbank, primary_board, diff_category)
                    ),
                }
            )

        self.persist_skillbank()
        skill_count_after = count_skills(self.skillbank)
        output_record = {
            "task_id": task_id,
            "source_experiment_name": self.source_experiment_name,
            "interaction_count": len(interactions),
            "classified_interaction_count": len(classification_records),
            "curated_interaction_count": sum(
                1 for item in interaction_outputs if not item.get("skipped")
            ),
            "skill_count_before": skill_count_before,
            "skill_count_after": skill_count_after,
            "trained_skillbank_file_path": self.trained_skillbank_file_path,
            "interaction_outputs": interaction_outputs,
        }
        output_path = os.path.join(task_output_logs_dir, "prediction_diff_curation.json")
        write_json(output_record, output_path, silent=True)
        print(
            f"[prediction_diff_curation] task={task_id} "
            f"skill_count={skill_count_after} output={output_path}"
        )
        return output_record

    def render_curator_prompt(
        self,
        task_context: str,
        previous_trajectory: str,
        current_interaction: str,
        current_classification: str,
        current_skill_bucket: str,
        next_interaction: str,
    ) -> str:
        return self.curator_prompt_template.format(
            task_context=task_context,
            previous_trajectory=previous_trajectory,
            current_interaction=current_interaction,
            current_classification=current_classification,
            current_skill_bucket=current_skill_bucket,
            next_interaction=next_interaction,
        )

    def normalize_curator_response(
        self,
        raw_response: str,
        current_bucket: list[dict[str, Any]],
    ) -> dict[str, Any]:
        parsed_response = extract_json_from_text(raw_response, "operations") or {}
        parse_errors: list[str] = []

        reasoning = str(parsed_response.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = "Curator returned empty reasoning."
            parse_errors.append("empty reasoning")

        raw_operations = parsed_response.get("operations")
        if raw_operations is None:
            raw_operations = []
        if not isinstance(raw_operations, list):
            parse_errors.append("operations is not a list")
            raw_operations = []

        current_bucket_ids = {
            str(skill.get("skill_id") or "").strip()
            for skill in current_bucket
            if isinstance(skill, dict)
        }
        normalized_operations: list[dict[str, Any]] = []
        for index, operation in enumerate(raw_operations):
            if not isinstance(operation, dict):
                parse_errors.append(f"operation {index} is not an object")
                continue

            operation_type = str(operation.get("type", "")).strip().upper()
            if operation_type == "ADD":
                skill = operation.get("skill")
                if not isinstance(skill, dict):
                    parse_errors.append(f"operation {index} missing skill object")
                    continue
                content = str(skill.get("content", "")).strip()
                note = str(skill.get("note", "")).strip()
                if not content:
                    parse_errors.append(f"operation {index} has empty skill.content")
                    continue
                normalized_operations.append(
                    {
                        "type": "ADD",
                        "skill": {
                            "content": content,
                            "note": note,
                        },
                    }
                )
                continue

            if operation_type == "MODIFY":
                target_skill_id = str(operation.get("target_skill_id", "")).strip()
                updated_skill = operation.get("updated_skill")
                if not target_skill_id:
                    parse_errors.append(f"operation {index} missing target_skill_id")
                    continue
                if target_skill_id not in current_bucket_ids:
                    parse_errors.append(
                        f"operation {index} references unknown target_skill_id={target_skill_id}"
                    )
                    continue
                if not isinstance(updated_skill, dict):
                    parse_errors.append(f"operation {index} missing updated_skill object")
                    continue
                content = str(updated_skill.get("content", "")).strip()
                note = str(updated_skill.get("note", "")).strip()
                if not content:
                    parse_errors.append(f"operation {index} has empty updated_skill.content")
                    continue
                normalized_operations.append(
                    {
                        "type": "MODIFY",
                        "target_skill_id": target_skill_id,
                        "updated_skill": {
                            "content": content,
                            "note": note,
                        },
                    }
                )
                continue

            parse_errors.append(f"operation {index} has unsupported type={operation_type}")

        return {
            "reasoning": reasoning,
            "operations": normalized_operations,
            "parse_errors": parse_errors,
        }

    def format_previous_trajectory(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return "(no previous trajectory)"

        chunks: list[str] = []
        for entry in history:
            lines = [
                f"[Interaction {entry.get('interaction_index')}]",
                "Action:",
                "```python",
                self.clip_text(str(entry.get("code") or ""), max_chars=2000),
                "```",
                "",
                "Actual Observation:",
                "```",
                self.clip_text(str(entry.get("actual_observation") or ""), max_chars=2500),
                "```",
            ]
            chunks.append("\n".join(lines))
        return self.clip_text("\n\n".join(chunks), max_chars=self.max_history_chars)

    def format_current_interaction(self, interaction: dict[str, Any]) -> str:
        lines = [
            f"Interaction Index: {interaction.get('interaction_index')}",
            "Current Reasoning:",
            "```",
            self.clip_text(str(interaction.get("current_reasoning") or ""), max_chars=2500),
            "```",
            "",
            "Current Action:",
            "```python",
            self.clip_text(str(interaction.get("current_code") or ""), max_chars=4000),
            "```",
            "",
            "Predicted Observation:",
            "```",
            self.clip_text(str(interaction.get("predicted_observation") or ""), max_chars=3500),
            "```",
            "",
            "Actual Observation:",
            "```",
            self.clip_text(str(interaction.get("actual_observation") or ""), max_chars=3500),
            "```",
        ]
        return "\n".join(lines)

    def format_current_classification(self, classification_record: dict[str, Any]) -> str:
        classifier_reason = classification_record.get("reason") or classification_record.get("reasoning") or ""
        lines = [
            f"Primary Board: {classification_record.get('primary_board', '')}",
            f"Diff Category: {classification_record.get('diff_category', '')}",
            "Classifier Reasoning:",
            "```",
            self.clip_text(str(classifier_reason), max_chars=1800),
            "```",
        ]
        return "\n".join(lines)

    def format_current_skill_bucket(
        self,
        primary_board: str,
        diff_category: str,
        current_bucket: list[dict[str, Any]],
    ) -> str:
        payload = {
            "primary_board": primary_board,
            "diff_category": diff_category,
            "skills": current_bucket,
        }
        return self.clip_text(json.dumps(payload, ensure_ascii=False, indent=2), max_chars=16000)

    def format_next_interaction(
        self,
        next_interaction: dict[str, Any] | None,
        next_classification: dict[str, Any] | None,
    ) -> str:
        if not next_interaction:
            return "(not provided)"

        payload = {
            "interaction_index": next_interaction.get("interaction_index"),
            "reasoning": self.clip_text(
                str(next_interaction.get("current_reasoning") or ""),
                max_chars=2000,
            ),
            "action": self.clip_text(
                str(next_interaction.get("current_code") or ""),
                max_chars=3000,
            ),
            "predicted_observation": self.clip_text(
                str(next_interaction.get("predicted_observation") or ""),
                max_chars=2500,
            ),
            "actual_observation": self.clip_text(
                str(next_interaction.get("actual_observation") or ""),
                max_chars=2500,
            ),
            "classification": {
                "primary_board": next_classification.get("primary_board") if next_classification else "",
                "diff_category": next_classification.get("diff_category") if next_classification else "",
                "reasoning": self.clip_text(
                    str(
                        next_classification.get("reason")
                        or next_classification.get("reasoning")
                        or ""
                    ) if next_classification else "",
                    max_chars=1200,
                ),
            },
        }
        return self.clip_text(json.dumps(payload, ensure_ascii=False, indent=2), max_chars=12000)

    def clip_text(self, text: str, max_chars: int | None = None) -> str:
        max_chars = max_chars if max_chars is not None else self.max_field_chars
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[TRUNCATED]"

    def persist_skillbank(self) -> None:
        os.makedirs(os.path.dirname(self.trained_skillbank_file_path), exist_ok=True)
        write_json(self.skillbank, self.trained_skillbank_file_path, silent=True)

    def save_skillbank_snapshot(self) -> None:
        if not self.trained_skillbank_file_path:
            raise ValueError("trained_skillbank_file_path is not set")
        snapshot_file_path = (
            self.trained_skillbank_file_path.rsplit(".json", 1)[0]
            + f"_{self.current_task_index + 1}.json"
        )
        write_json(self.skillbank, snapshot_file_path, silent=True)
        print(
            f"Saved skillbank snapshot at task {self.current_task_index + 1}: "
            f"{snapshot_file_path}"
        )

    def _source_logs_dir(self, task_id: str) -> str:
        return os.path.join(
            path_store.experiment_outputs,
            self.source_experiment_name,
            "tasks",
            task_id,
            "logs",
        )

    def _target_task_logs_dir(self, task_id: str) -> str:
        return os.path.join(
            path_store.experiment_outputs,
            self.source_experiment_name,
            "tasks",
            task_id,
            "logs",
        )

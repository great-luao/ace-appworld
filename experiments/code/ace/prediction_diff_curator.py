import os
import re
from typing import Any

from appworld import AppWorld
from appworld.common.path_store import path_store
from appworld.common.utils import FromDict, read_file, write_json, yield_jsonl
from appworld.task import Task
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.playbook import (
    apply_curator_operations,
    extract_json_from_text,
    get_next_global_id,
    get_playbook_stats,
)


PREDICTION_TRIGGER_TEXT = "Before writing your next code block, predict what the environment would return"
ALLOWED_SECTIONS = {
    "strategies_and_hard_rules",
    "apis_to_use_for_specific_information",
    "useful_code_snippets_and_templates",
    "common_mistakes_and_correct_strategies",
    "problem_solving_heuristics_and_workflows",
    "verification_checklist",
    "troubleshooting_and_pitfalls",
    "others",
}


class PredictionDiffCurator(FromDict):
    def __init__(
        self,
        curator_model_config: dict,
        curator_prompt_file_path: str,
        initial_playbook_file_path: str,
        trained_playbook_file_path: str,
        source_experiment_name: str,
        classification_file_name: str = "prediction_diff_classification.jsonl",
        max_field_chars: int = 10000,
        max_classification_chars: int = 50000,
        max_history_chars: int = 100000,
        log_lm_calls: bool = True,
        prediction_trigger_text: str = PREDICTION_TRIGGER_TEXT,
    ):
        self.curator_model_config = curator_model_config
        self.curator_model: LiteLLMGenerator | None = None
        self.curator_prompt_file_path = curator_prompt_file_path
        self.curator_prompt_template = read_file(curator_prompt_file_path.replace("/", os.sep))
        self.initial_playbook_file_path = initial_playbook_file_path
        self.trained_playbook_file_path = trained_playbook_file_path
        self.source_experiment_name = source_experiment_name
        self.classification_file_name = classification_file_name
        self.max_field_chars = max_field_chars
        self.max_classification_chars = max_classification_chars
        self.max_history_chars = max_history_chars
        self.log_lm_calls = log_lm_calls
        self.prediction_trigger_text = prediction_trigger_text
        self.playbook = ""
        self.next_global_id = 1
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
                "it updates one shared playbook across tasks."
            )

        self.initialize_playbook()
        print(
            f"[prediction_diff_curation] source_experiment={self.source_experiment_name} "
            f"tasks={len(task_ids)} playbook={self.trained_playbook_file_path}"
        )
        for task_index, task_id in enumerate(task_ids):
            self.current_task_index = task_index
            self.curate_task(task_id)
            if (self.current_task_index + 1) % 30 == 0:
                self.save_playbook_snapshot()

    def initialize_playbook(self) -> None:
        if os.path.exists(self.initial_playbook_file_path):
            self.playbook = read_file(self.initial_playbook_file_path.replace("/", os.sep))
        else:
            self.playbook = "(empty)"
        self.next_global_id = get_next_global_id(self.playbook)
        self.persist_playbook()

    def curate_task(self, task_id: str) -> dict[str, Any] | None:
        source_logs_dir = self._source_logs_dir(task_id)
        environment_log_path = os.path.join(source_logs_dir, "environment_io.md")
        lm_calls_path = os.path.join(source_logs_dir, "lm_calls.jsonl")
        classification_path = os.path.join(source_logs_dir, self.classification_file_name)

        required_paths = [environment_log_path, lm_calls_path, classification_path]
        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        if missing_paths:
            print(f"[prediction_diff_curation] skip task={task_id}, missing files: {missing_paths}")
            return None

        classification_records = list(yield_jsonl(classification_path))
        if not classification_records:
            print(
                f"[prediction_diff_curation] skip task={task_id}, empty classification file: "
                f"{classification_path}"
            )
            return None

        environment_entries = AppWorld.parse_environment_io_log(file_path=environment_log_path)
        lm_calls = list(yield_jsonl(lm_calls_path))
        reconstructed_steps = self.extract_reconstructed_steps(lm_calls)
        interactions = self.build_reconstructed_interactions(
            environment_entries=environment_entries,
            reconstructed_steps=reconstructed_steps,
        )

        task = Task.load(task_id=task_id)
        question_context = getattr(task, "instruction", "")
        classification_results = self.format_classification_results(classification_records)
        conversation_history = self.format_full_conversation_history(interactions)
        prompt = self.render_curator_prompt(
            question_context=question_context,
            classification_results=classification_results,
            conversation_history=conversation_history,
        )

        task_output_logs_dir = self._target_task_logs_dir(task_id)
        os.makedirs(task_output_logs_dir, exist_ok=True)
        if self.log_lm_calls:
            curator_lm_calls_path = os.path.join(task_output_logs_dir, "curator_lm_calls.jsonl")
            self._get_curator_model().log_calls_to(file_path=curator_lm_calls_path)

        playbook_stats_before = get_playbook_stats(self.playbook)
        model_output = self._get_curator_model().generate(
            messages=[{"role": "user", "content": prompt}]
        )
        raw_response = model_output.get("content") or ""
        normalized = self.normalize_curator_response(raw_response)

        operations = normalized["operations"]
        if operations:
            self.playbook, self.next_global_id = apply_curator_operations(
                self.playbook,
                operations,
                self.next_global_id,
            )
        self.persist_playbook()
        playbook_stats_after = get_playbook_stats(self.playbook)

        output_record = {
            "task_id": task_id,
            "source_experiment_name": self.source_experiment_name,
            "classification_record_count": len(classification_records),
            "interaction_count": len(interactions),
            "reasoning": normalized["reasoning"],
            "operations": operations,
            "applied_operation_count": len(operations),
            "model_response_raw": raw_response,
            "parse_errors": normalized["parse_errors"],
            "playbook_total_bullets_before": playbook_stats_before.get("total_bullets"),
            "playbook_total_bullets_after": playbook_stats_after.get("total_bullets"),
            "trained_playbook_file_path": self.trained_playbook_file_path,
        }
        output_path = os.path.join(task_output_logs_dir, "prediction_diff_curation.json")
        write_json(output_record, output_path, silent=True)
        print(
            f"[prediction_diff_curation] task={task_id} operations={len(operations)} "
            f"output={output_path}"
        )
        return output_record

    def render_curator_prompt(
        self,
        question_context: str,
        classification_results: str,
        conversation_history: str,
    ) -> str:
        prompt = self.curator_prompt_template.format(
            question_context=question_context,
            current_playbook=self.playbook,
            guidebook=classification_results,
        )
        return prompt + "\n\n=== FULL CONVERSATION HISTORY ===\n" + conversation_history

    def normalize_curator_response(self, raw_response: str) -> dict[str, Any]:
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

        normalized_operations: list[dict[str, str]] = []
        for index, operation in enumerate(raw_operations):
            if not isinstance(operation, dict):
                parse_errors.append(f"operation {index} is not an object")
                continue
            operation_type = str(operation.get("type", "")).strip()
            section = str(operation.get("section", "")).strip()
            content = str(operation.get("content", "")).strip()
            if operation_type != "ADD":
                parse_errors.append(f"operation {index} has unsupported type={operation_type}")
                continue
            normalized_section = self.normalize_section(section)
            if normalized_section not in ALLOWED_SECTIONS:
                parse_errors.append(
                    f"operation {index} has invalid section={section} "
                    f"(normalized={normalized_section})"
                )
                continue
            if not content:
                parse_errors.append(f"operation {index} has empty content")
                continue
            normalized_operations.append(
                {
                    "type": "ADD",
                    "section": section,
                    "content": content,
                }
            )

        return {
            "reasoning": reasoning,
            "operations": normalized_operations,
            "parse_errors": parse_errors,
        }

    def format_classification_results(self, classification_records: list[dict[str, Any]]) -> str:
        if not classification_records:
            return "(no classification results)"

        chunks: list[str] = []
        for record in sorted(
            classification_records,
            key=lambda item: int(item.get("interaction_index") or 0),
        ):
            evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
            chunk = "\n".join(
                [
                    f"[Interaction {record.get('interaction_index')}]",
                    f"primary_board: {record.get('primary_board', '')}",
                    f"diff_category: {record.get('diff_category', '')}",
                    "classifier_reasoning:",
                    self.clip_text(str(record.get("reasoning") or ""), max_chars=1200),
                    "evidence.current_reasoning_summary:",
                    self.clip_text(str(evidence.get("current_reasoning_summary") or ""), max_chars=500),
                    "evidence.key_predicted_claim:",
                    self.clip_text(str(evidence.get("key_predicted_claim") or ""), max_chars=700),
                    "evidence.key_actual_fact:",
                    self.clip_text(str(evidence.get("key_actual_fact") or ""), max_chars=700),
                ]
            )
            chunks.append(chunk)
        return self.clip_text("\n\n".join(chunks), max_chars=self.max_classification_chars)

    def format_full_conversation_history(self, interactions: list[dict[str, Any]]) -> str:
        if not interactions:
            return "(no reconstructed trajectory)"

        chunks: list[str] = []
        for interaction in interactions:
            reasoning = str(interaction.get("current_reasoning") or "").strip()
            code = str(interaction.get("current_code") or "")
            actual_output = self.clip_text(str(interaction.get("actual_output") or ""))

            lines = [f"[Interaction {interaction['interaction_index']}]"]
            if reasoning:
                lines.extend(
                    [
                        "ASSISTANT:",
                        self.clip_text(reasoning, max_chars=2000),
                        "",
                    ]
                )
            lines.extend(
                [
                    "Code:",
                    "```python",
                    code,
                    "```",
                    "",
                    "USER:",
                    "Output:",
                    "```",
                    actual_output,
                    "```",
                ]
            )
            chunks.append("\n".join(lines))

        return self.clip_text("\n\n".join(chunks), max_chars=self.max_history_chars)

    def build_reconstructed_interactions(
        self,
        environment_entries: list[dict[str, str]],
        reconstructed_steps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        interactions: list[dict[str, Any]] = []
        step_cursor = 0

        for index, environment_entry in enumerate(environment_entries):
            current_code = environment_entry.get("input") or ""
            current_actual_output = environment_entry.get("output") or ""

            matched_step, step_cursor = self.match_step_for_code(
                current_code=current_code,
                reconstructed_steps=reconstructed_steps,
                step_cursor=step_cursor,
            )
            current_reasoning = matched_step.get("reasoning") if matched_step else ""
            interactions.append(
                {
                    "interaction_index": index + 1,
                    "current_reasoning": current_reasoning,
                    "current_code": current_code,
                    "actual_output": current_actual_output,
                }
            )
        return interactions

    def match_step_for_code(
        self,
        current_code: str,
        reconstructed_steps: list[dict[str, Any]],
        step_cursor: int,
        lookahead: int = 4,
    ) -> tuple[dict[str, Any] | None, int]:
        normalized_current = self.normalize_code(current_code)
        upper = min(len(reconstructed_steps), step_cursor + lookahead)

        for candidate_index in range(step_cursor, upper):
            candidate_code = self.normalize_code(reconstructed_steps[candidate_index].get("code") or "")
            if candidate_code == normalized_current:
                return reconstructed_steps[candidate_index], candidate_index + 1

        if step_cursor < len(reconstructed_steps):
            return reconstructed_steps[step_cursor], step_cursor + 1
        return None, step_cursor

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

            steps.append(
                {
                    "lm_call_index": call_index,
                    "code": code,
                    "reasoning": reasoning,
                }
            )
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

    def normalize_code(self, code: str) -> str:
        lines = [line.rstrip() for line in (code or "").strip().splitlines()]
        return "\n".join(lines).strip()

    def clip_text(self, text: str, max_chars: int | None = None) -> str:
        max_chars = max_chars if max_chars is not None else self.max_field_chars
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[TRUNCATED]"

    def normalize_section(self, section: str) -> str:
        return section.strip().lower().replace(" ", "_").replace("&", "and").rstrip(":")

    def persist_playbook(self) -> None:
        os.makedirs(os.path.dirname(self.trained_playbook_file_path), exist_ok=True)
        with open(self.trained_playbook_file_path, "w", encoding="utf-8") as file:
            file.write(self.playbook)

    def save_playbook_snapshot(self) -> None:
        if not self.trained_playbook_file_path:
            raise ValueError("trained_playbook_file_path is not set")
        snapshot_file_path = (
            self.trained_playbook_file_path.split(".txt")[0]
            + str(self.current_task_index + 1)
            + ".txt"
        )
        with open(snapshot_file_path, "w", encoding="utf-8") as file:
            file.write(self.playbook)
        print(
            f"Saved playbook snapshot at task {self.current_task_index + 1}: "
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

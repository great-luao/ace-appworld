import json
import os
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from appworld import AppWorld
from appworld.common.constants import DEFAULT_EXPERIMENT_NAME, SINGLE_HORIZONTAL_RULE
from appworld.common.utils import read_json
from appworld_experiments.code.ace.base_agent import BaseAgent, ExecutionIO
from appworld_experiments.code.ace.base_react import BaseSimplifiedReActAgent
from appworld_experiments.code.ace.prediction_diff_retrieval import (
    PredictionDiffRetrievalClassifier,
)
from appworld_experiments.code.ace.skillbank import ensure_skillbank_shape, get_bucket


DEFAULT_RETRIEVAL_MIN_CLASS_SCORE = 0.15
ROLLBACK_BOARDS = {"docs_lookup", "read_fetch"}
POST_STATE_BOARDS = {"auth", "local_reasoning"}


@BaseAgent.register("prediction_diff_inference_react")
class PredictionDiffInferenceReActAgent(BaseSimplifiedReActAgent):
    def __init__(
        self,
        retrieval_datapoints_file_path: str,
        skillbank_file_path: str,
        retrieval_backend_name: str = "hybrid_tfidf",
        retrieval_top_k: int = 5,
        retrieval_evidence_top_n: int = 5,
        retrieval_min_class_score: float | None = DEFAULT_RETRIEVAL_MIN_CLASS_SCORE,
        skill_max_per_injection: int = 5,
        max_rollbacks_per_step: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retrieval_datapoints_file_path = retrieval_datapoints_file_path
        self.skillbank_file_path = skillbank_file_path
        self.retrieval_backend_name = retrieval_backend_name
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_evidence_top_n = retrieval_evidence_top_n
        self.retrieval_min_class_score = retrieval_min_class_score
        self.skill_max_per_injection = skill_max_per_injection
        self.max_rollbacks_per_step = max_rollbacks_per_step

        self.retrieval_classifier = PredictionDiffRetrievalClassifier.from_datapoints_file(
            datapoints_path=retrieval_datapoints_file_path.replace("/", os.sep),
            backend_name=retrieval_backend_name,
            top_k=retrieval_top_k,
            evidence_top_n=retrieval_evidence_top_n,
            min_class_score=retrieval_min_class_score,
        )
        self.skillbank = ensure_skillbank_shape(read_json(skillbank_file_path.replace("/", os.sep)))
        self.guided_environment_io: list[dict[str, str]] = []
        self.guided_predicted_environment_io: list[dict[str, str]] = []
        self.comparison_environment_io: list[dict[str, str]] = []
        self.comparison_predicted_environment_io: list[dict[str, str]] = []
        self.skill_guidance_history: list[dict[str, str]] = []
        self.pending_skill_guidance = ""
        self.latest_skill_guidance = ""

    def initialize(self, world: AppWorld):
        super().initialize(world)
        self.guided_environment_io = []
        self.guided_predicted_environment_io = []
        self.comparison_environment_io = []
        self.comparison_predicted_environment_io = []
        self.skill_guidance_history = []
        self.pending_skill_guidance = ""
        self.latest_skill_guidance = ""
        self._save_comparison_io_logs()
        self._save_guided_environment_io_logs()
        self._save_skill_guidance_history_log()

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list[ExecutionIO]
    ) -> tuple[ExecutionIO, float]:
        transient_guidance_index = None
        transient_guidance = ""
        if self.pending_skill_guidance:
            transient_guidance = self.pending_skill_guidance
            transient_guidance_index = len(self.messages)
            self.messages.append({"role": "user", "content": transient_guidance})
            self.pending_skill_guidance = ""

        try:
            return super().next_execution_inputs_and_cost(last_execution_outputs)
        finally:
            if transient_guidance_index is not None:
                self.remove_message_at_index(transient_guidance_index)
                if self.latest_skill_guidance == transient_guidance:
                    self.latest_skill_guidance = ""

    def solve_task(self, task_id: str, experiment_name: str | None = None):
        experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self.cost_tracker.reset(task_id)
        with AppWorld(
            task_id=task_id, experiment_name=experiment_name, **self.appworld_config
        ) as world:
            self.initialize(world)
            for _ in range(self.max_steps):
                self.step_number += 1
                rollback_count = 0
                regenerate_current_step = True

                while regenerate_current_step:
                    regenerate_current_step = False
                    checkpoint_id = world.save_state(
                        f"pre_step_{self.step_number}_attempt_{rollback_count + 1}"
                    )
                    namespace_snapshot = self.snapshot_python_namespace(world)
                    execution_inputs, cost = self.next_execution_inputs_and_cost([])
                    if len(execution_inputs) != 1:
                        raise ValueError("prediction_diff_inference_react expects one execution per step.")

                    execution_input = execution_inputs[0]
                    execution_output = ExecutionIO(
                        content=world.execute(execution_input.content),
                        metadata=execution_input.metadata,
                    )
                    predicted_output = execution_input.metadata.get("predicted_output", "")
                    self.record_predicted_environment_io(
                        input_code=execution_input.content,
                        predicted_output=predicted_output,
                        actual_output=execution_output.content,
                    )
                    interaction_index = len(self.world.environment_io) - 1

                    self.cost_tracker.add(task_id, cost)
                    self.log_cost()

                    decision = self.build_injection_decision(
                        execution_input=execution_input,
                        execution_output=execution_output,
                        checkpoint_id=checkpoint_id,
                        rollback_count=rollback_count,
                    )
                    if (
                        decision["injection_mode"] == "rollback"
                        and rollback_count >= self.max_rollbacks_per_step
                    ):
                        decision["rollback_skipped_reason"] = "max_rollbacks_per_step_reached"
                        decision["injection_mode"] = "none"
                    self.record_skill_guidance_interaction(decision)
                    self.log_skill_injection_event(decision)

                    if decision["injection_mode"] == "rollback":
                        self.record_comparison_interaction(interaction_index)
                        self.log_discarded_rollback_interaction(decision)
                        world.load_state(checkpoint_id)
                        self.restore_python_namespace(world, namespace_snapshot)
                        self.remove_last_assistant_message()
                        self.append_skill_guidance_message(decision)
                        rollback_count += 1
                        regenerate_current_step = True
                        if self.cost_tracker.exceeded():
                            break
                        continue

                    self.accept_execution_interaction(
                        execution_input=execution_input,
                        execution_output=execution_output,
                    )
                    self.record_comparison_interaction(interaction_index)
                    if decision["injection_mode"] == "post_state":
                        self.append_skill_guidance_message(decision)

                    if world.task_completed() or self.cost_tracker.exceeded():
                        break

                if world.task_completed() or self.cost_tracker.exceeded():
                    break
        self.logger.complete_task()

    def build_injection_decision(
        self,
        execution_input: ExecutionIO,
        execution_output: ExecutionIO,
        checkpoint_id: str,
        rollback_count: int,
    ) -> dict[str, Any]:
        code = execution_input.content
        predicted_output = execution_input.metadata.get("predicted_output", "")
        base_event = {
            "step_number": self.step_number,
            "checkpoint_id": checkpoint_id,
            "rollback_count_before": rollback_count,
            "current_code": code,
            "predicted_output": predicted_output,
            "actual_output": execution_output.content,
            "classification": None,
            "selected_skills": [],
            "injection_mode": "none",
        }
        if not code.strip() or self.is_complete_task_action(code):
            return base_event

        classification = self.retrieval_classifier.classify(
            {
                "current_code": code,
                "predicted_output": predicted_output,
                "actual_output": execution_output.content,
            }
        )
        base_event["classification"] = classification
        if not classification["should_retrieve_skill"]:
            return base_event

        primary_board = classification["predicted_board"]
        diff_category = classification["predicted_diff_category"]
        selected_skills = self.select_skills(
            primary_board=primary_board,
            diff_category=diff_category,
            current_code=code,
            predicted_output=predicted_output,
            actual_output=execution_output.content,
        )
        base_event["selected_skills"] = selected_skills
        if not selected_skills:
            return base_event

        if primary_board in ROLLBACK_BOARDS:
            base_event["injection_mode"] = "rollback"
        elif primary_board in POST_STATE_BOARDS:
            base_event["injection_mode"] = "post_state"
        return base_event

    def select_skills(
        self,
        primary_board: str,
        diff_category: str,
        current_code: str,
        predicted_output: str,
        actual_output: str,
    ) -> list[dict[str, Any]]:
        bucket = get_bucket(self.skillbank, primary_board, diff_category)
        if len(bucket) <= self.skill_max_per_injection:
            return [self.format_selected_skill(skill, score=None) for skill in bucket]

        query_text = "\n\n".join([current_code, predicted_output, actual_output])
        skill_texts = [
            "\n".join([skill.get("content", ""), skill.get("note", "")])
            for skill in bucket
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        skill_matrix = vectorizer.fit_transform(skill_texts)
        query_vector = vectorizer.transform([query_text])
        scores = linear_kernel(query_vector, skill_matrix).ravel()
        ranked_indices = sorted(
            range(len(bucket)),
            key=lambda index: (-float(scores[index]), bucket[index].get("skill_id", "")),
        )
        selected = []
        for index in ranked_indices[: self.skill_max_per_injection]:
            selected.append(self.format_selected_skill(bucket[index], score=float(scores[index])))
        return selected

    def format_selected_skill(self, skill: dict[str, Any], score: float | None) -> dict[str, Any]:
        return {
            "skill_id": skill.get("skill_id", ""),
            "content": skill.get("content", ""),
            "note": skill.get("note", ""),
            "source": skill.get("source", {}),
            "selection_score": score,
        }

    def accept_execution_interaction(
        self,
        execution_input: ExecutionIO,
        execution_output: ExecutionIO,
    ) -> None:
        self.logger.show_message(
            role="environment",
            message=execution_output.content,
            step_number=self.step_number,
        )
        self.messages.append(
            {
                "role": "user",
                "content": "Output:\n```\n" + execution_output.content + "```\n\n",
            }
        )
        self.guided_environment_io.append(
            {"input": execution_input.content, "output": execution_output.content.rstrip()}
        )
        predicted_output = execution_input.metadata.get("predicted_output", "") or ""
        self.guided_predicted_environment_io.append(
            {"input": execution_input.content, "output": predicted_output.rstrip()}
        )
        self._save_guided_environment_io_logs()

    def record_comparison_interaction(self, interaction_index: int) -> None:
        expected_index = self.step_number - 1
        if len(self.comparison_environment_io) != expected_index:
            return
        self.comparison_environment_io.append(self.world.environment_io[interaction_index])
        self.comparison_predicted_environment_io.append(
            self.predicted_environment_io[interaction_index]
        )
        self._save_comparison_io_logs()

    def record_skill_guidance_interaction(self, decision: dict[str, Any]) -> None:
        expected_index = self.step_number - 1
        if len(self.skill_guidance_history) != expected_index:
            return
        self.skill_guidance_history.append(
            {
                "input": decision.get("current_code", "").rstrip(),
                "output": self.format_skill_guidance_history_entry(decision),
            }
        )
        self._save_skill_guidance_history_log()

    def format_skill_guidance_history_entry(self, decision: dict[str, Any]) -> str:
        classification = decision.get("classification") or {}
        selected_skills = decision.get("selected_skills") or []
        lines = [
            "classification:",
            f"board: {classification.get('predicted_board') or '(none)'}",
            f"diff_category: {classification.get('predicted_diff_category') or '(none)'}",
            "skills:",
        ]
        if not selected_skills:
            lines.append("(No skill retrieved)")
        else:
            for skill in selected_skills:
                lines.append(f"[{skill['skill_id']}] {skill['content']}")
        return "\n".join(lines).rstrip()

    def append_skill_guidance_message(self, decision: dict[str, Any]) -> None:
        guidance = self.build_skill_guidance_message(decision)
        self.pending_skill_guidance = guidance
        self.latest_skill_guidance = guidance

    def build_skill_guidance_message(self, decision: dict[str, Any]) -> str:
        lines = [
            "Retrieved Skill Guidance:",
            "Use the following skill guidance for your next code block only.",
            "Do not mention or quote this guidance in your response; just use it to choose the next action.",
            "Verify API names, parameters, schemas, and values against actual API docs and environment outputs.",
        ]
        lines.append("")
        for skill in decision["selected_skills"]:
            lines.append(f"[{skill['skill_id']}] {skill['content']}")
        return "\n".join(lines).rstrip() + "\n\n"

    def remove_message_at_index(self, index: int) -> None:
        if index >= len(self.messages):
            return
        if self.messages[index].get("content") == self.latest_skill_guidance:
            self.messages.pop(index)

    def remove_last_assistant_message(self) -> None:
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index]["role"] == "assistant":
                self.messages.pop(index)
                return
        raise ValueError("No assistant message available to remove during rollback.")

    def snapshot_python_namespace(self, world: AppWorld) -> dict[str, Any]:
        if not hasattr(world, "shell"):
            return {}
        return dict(world.shell.user_ns)

    def restore_python_namespace(self, world: AppWorld, snapshot: dict[str, Any]) -> None:
        if not snapshot or not hasattr(world, "shell"):
            return
        protected_keys = {"apis", "requester"}
        current_namespace = world.shell.user_ns
        for key in list(current_namespace):
            if key not in snapshot and key not in protected_keys:
                current_namespace.pop(key, None)
        for key, value in snapshot.items():
            if key not in protected_keys:
                current_namespace[key] = value

    def log_skill_injection_event(self, event: dict[str, Any]) -> None:
        self.append_jsonl("skill_injection_events.jsonl", event)

    def log_discarded_rollback_interaction(self, event: dict[str, Any]) -> None:
        self.append_jsonl("discarded_rollback_interactions.jsonl", event)

    def append_jsonl(self, file_name: str, payload: dict[str, Any]) -> None:
        file_path = os.path.join(self.world.output_logs_directory, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_guided_environment_io_logs(self) -> None:
        self._write_io_markdown("guided_environment_io.md", self.guided_environment_io)
        self._write_io_markdown(
            "guided_predicted_environment_io.md",
            self.guided_predicted_environment_io,
        )

    def _save_comparison_io_logs(self) -> None:
        self._write_io_markdown("environment_io.md", self.comparison_environment_io)
        self._write_io_markdown(
            "predicted_environment_io.md",
            self.comparison_predicted_environment_io,
        )

    def _save_skill_guidance_history_log(self) -> None:
        file_path = os.path.join(self.world.output_logs_directory, "active_skill_guidance.md")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not self.skill_guidance_history:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("Retrieved Skill Guidance:\n(No skill guidance interactions yet)\n")
            return
        self._write_io_markdown("active_skill_guidance.md", self.skill_guidance_history)

    def _write_io_markdown(self, file_name: str, entries: list[dict[str, str]]) -> None:
        file_path = os.path.join(self.world.output_logs_directory, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            for index, entry in enumerate(entries):
                content = "\n".join(
                    [
                        f"\n### Environment Interaction {index + 1}\n{SINGLE_HORIZONTAL_RULE}",
                        f"```python\n{entry['input']}\n```\n",
                        f"```\n{entry['output']}\n```\n\n",
                    ]
                )
                file.write(content)

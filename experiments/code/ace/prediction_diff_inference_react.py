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
from appworld_experiments.code.ace.skillbank import (
    PRIMARY_BOARDS,
    ensure_skillbank_shape,
    get_bucket,
)


DEFAULT_RETRIEVAL_MIN_CLASS_SCORE = 0.15
DEFAULT_SKILL_SELECTION_MODE = "multi_bucket"
SKILL_SELECTION_MODES = {"multi_bucket"}
PRIMARY_BOARD_TITLES = {
    "docs_lookup": "DOCS LOOKUP",
    "auth": "AUTH",
    "read_fetch": "READ FETCH",
    "local_reasoning": "LOCAL REASONING",
}


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
        skill_selection_mode: str = DEFAULT_SKILL_SELECTION_MODE,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retrieval_datapoints_file_path = retrieval_datapoints_file_path
        self.skillbank_file_path = skillbank_file_path
        self.retrieval_backend_name = retrieval_backend_name
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_evidence_top_n = retrieval_evidence_top_n
        self.retrieval_min_class_score = retrieval_min_class_score
        self.skill_selection_mode = skill_selection_mode
        if self.skill_selection_mode not in SKILL_SELECTION_MODES:
            raise ValueError("skill_selection_mode must be 'multi_bucket'.")

        self.retrieval_classifier = PredictionDiffRetrievalClassifier.from_datapoints_file(
            datapoints_path=retrieval_datapoints_file_path.replace("/", os.sep),
            backend_name=retrieval_backend_name,
            top_k=retrieval_top_k,
            evidence_top_n=retrieval_evidence_top_n,
            min_class_score=retrieval_min_class_score,
        )
        self.skillbank = ensure_skillbank_shape(read_json(skillbank_file_path.replace("/", os.sep)))
        self.skill_guidance_history: list[dict[str, str]] = []
        self.pending_skill_guidance = ""
        self.latest_skill_guidance = ""

    def initialize(self, world: AppWorld):
        super().initialize(world)
        self.skill_guidance_history = []
        self.pending_skill_guidance = ""
        self.latest_skill_guidance = ""
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
                execution_inputs, cost = self.next_execution_inputs_and_cost([])
                execution_inputs = [
                    execution_input
                    for execution_input in execution_inputs
                    if (execution_input.content or "").strip()
                ]
                if not execution_inputs:
                    self.cost_tracker.add(task_id, cost)
                    self.log_cost()
                    break
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
                self._save_predicted_environment_io_log()

                self.cost_tracker.add(task_id, cost)
                self.log_cost()

                decision = self.build_injection_decision(
                    execution_input=execution_input,
                    execution_output=execution_output,
                )
                self.record_skill_guidance_interaction(decision)
                self.log_skill_injection_event(decision)

                self.accept_execution_interaction(
                    execution_input=execution_input,
                    execution_output=execution_output,
                )
                if decision["injection_mode"] == "post_injection":
                    self.append_skill_guidance_message(decision)

                if world.task_completed() or self.cost_tracker.exceeded():
                    break
        self.logger.complete_task()

    def build_injection_decision(
        self,
        execution_input: ExecutionIO,
        execution_output: ExecutionIO,
    ) -> dict[str, Any]:
        code = execution_input.content
        predicted_output = execution_input.metadata.get("predicted_output", "")
        base_event = {
            "step_number": self.step_number,
            "current_code": code,
            "predicted_output": predicted_output,
            "actual_output": execution_output.content,
            "selection_mode": self.skill_selection_mode,
            "classification": None,
            "selection_classification": None,
            "policy_board": None,
            "selected_buckets": [],
            "selected_skills": [],
            "injection_mode": "none",
        }
        if not code.strip() or self.is_complete_task_action(code):
            return base_event

        query_datapoint = {
            "current_code": code,
            "predicted_output": predicted_output,
            "actual_output": execution_output.content,
        }
        classification = self.retrieval_classifier.classify_multi_bucket(query_datapoint)
        base_event["classification"] = classification
        primary_board = classification["predicted_board"]
        base_event["policy_board"] = primary_board
        base_event["selection_classification"] = classification
        selected_buckets = classification["selected_buckets"]
        base_event["selected_buckets"] = selected_buckets
        if not classification["should_retrieve_skill"]:
            return base_event

        selected_skills = self.select_skills(
            selected_buckets=selected_buckets,
            current_code=code,
            predicted_output=predicted_output,
            actual_output=execution_output.content,
        )
        base_event["selected_skills"] = selected_skills
        if not selected_skills:
            return base_event

        base_event["injection_mode"] = "post_injection"
        return base_event

    def select_skills(
        self,
        selected_buckets: list[dict[str, Any]],
        current_code: str,
        predicted_output: str,
        actual_output: str,
    ) -> list[dict[str, Any]]:
        candidate_entries = []
        for selected_bucket in selected_buckets:
            primary_board = selected_bucket["primary_board"]
            diff_category = selected_bucket["diff_category"]
            for skill in get_bucket(self.skillbank, primary_board, diff_category):
                candidate_entries.append((skill, selected_bucket))
        if not candidate_entries:
            return []

        query_text = "\n\n".join([current_code, predicted_output, actual_output])
        skill_texts = [
            "\n".join([skill.get("content", ""), skill.get("note", "")])
            for skill, _ in candidate_entries
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        skill_matrix = vectorizer.fit_transform(skill_texts)
        query_vector = vectorizer.transform([query_text])
        scores = linear_kernel(query_vector, skill_matrix).ravel()
        ranked_indices = sorted(
            range(len(candidate_entries)),
            key=lambda index: (
                -float(scores[index]),
                -float(candidate_entries[index][1]["bucket_score"]),
                candidate_entries[index][0].get("skill_id", ""),
            ),
        )
        selected = []
        for index in ranked_indices:
            skill, selected_bucket = candidate_entries[index]
            selected.append(
                self.format_selected_skill(
                    skill,
                    score=float(scores[index]),
                    bucket_score=float(selected_bucket["bucket_score"]),
                    board_score=float(selected_bucket["board_score"]),
                    category_score=float(selected_bucket["category_score"]),
                )
            )
        return selected

    def format_selected_skill(
        self,
        skill: dict[str, Any],
        score: float | None,
        bucket_score: float | None = None,
        board_score: float | None = None,
        category_score: float | None = None,
    ) -> dict[str, Any]:
        return {
            "skill_id": skill.get("skill_id", ""),
            "content": skill.get("content", ""),
            "note": skill.get("note", ""),
            "source": skill.get("source", {}),
            "selection_score": score,
            "bucket_score": bucket_score,
            "board_score": board_score,
            "category_score": category_score,
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
        selected_buckets = decision.get("selected_buckets") or []
        selected_skills = decision.get("selected_skills") or []
        lines = [
            f"selection_mode: {decision.get('selection_mode') or '(none)'}",
            f"policy_board: {decision.get('policy_board') or '(none)'}",
            "classification:",
            f"board: {classification.get('predicted_board') or '(none)'}",
            f"min_class_score: {classification.get('min_class_score')}",
            "selected_buckets:",
        ]
        if not selected_buckets:
            lines.append("(No bucket selected)")
        else:
            for bucket in selected_buckets:
                lines.append(
                    "{primary_board}/{diff_category}: bucket={bucket_score} "
                    "board={board_score} category={category_score}".format(**bucket)
                )
        lines.append(
            "skills:",
        )
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
            "Here are some Skill Guidances for you,",
            "you can take the useful part of them as an inspriation or referenc:.",
        ]
        lines.append("")
        lines.extend(self.build_playbook_style_guidance_lines(decision["selected_skills"]))
        return "\n".join(lines).rstrip() + "\n\n"

    def build_playbook_style_guidance_lines(
        self, selected_skills: list[dict[str, Any]]
    ) -> list[str]:
        grouped_skills: dict[str, list[dict[str, Any]]] = {}
        for skill in selected_skills:
            source = skill.get("source", {})
            primary_board = str(source.get("primary_board") or "unknown")
            grouped_skills.setdefault(primary_board, []).append(skill)

        lines: list[str] = []
        for primary_board in PRIMARY_BOARDS:
            skills = grouped_skills.get(primary_board, [])
            if not skills:
                continue
            if lines:
                lines.append("")
            lines.append(f"## {PRIMARY_BOARD_TITLES[primary_board]}")
            for skill in skills:
                lines.append(f"[{skill['skill_id']}] {skill['content']}")
        return lines

    def remove_message_at_index(self, index: int) -> None:
        if index >= len(self.messages):
            return
        if self.messages[index].get("content") == self.latest_skill_guidance:
            self.messages.pop(index)

    def log_skill_injection_event(self, event: dict[str, Any]) -> None:
        self.append_jsonl("skill_injection_events.jsonl", event)

    def append_jsonl(self, file_name: str, payload: dict[str, Any]) -> None:
        file_path = os.path.join(self.world.output_logs_directory, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

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

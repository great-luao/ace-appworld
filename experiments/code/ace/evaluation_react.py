import copy
import json
import os
import re
from typing import Any

from jinja2 import Template

from appworld import AppWorld
from appworld.common.utils import read_file
from appworld_experiments.code.ace.evaluation_agent import Agent, ExecutionIO
from appworld_experiments.code.ace.lite_llm_generator import LiteLLMGenerator
from appworld_experiments.code.ace.playbook import (
    build_playbook_bullet_lookup,
    extract_json_from_text,
    extract_playbook_bullets,
)


@Agent.register("ace_evaluation_react")
class SimplifiedReActAgent(Agent):
    def __init__(
        self,
        generator_prompt_file_path: str | None = None,
        trained_playbook_file_path: str | None = None,
        retrieval_prompt_file_path: str | None = None,
        retrieval_model_config: dict[str, Any] | None = None,
        retrieve_enabled: bool = False,
        retrieve_mode: str = "prefix_subset",
        retrieval_max_skills_per_call: int = 5,
        prefix_max_skills: int = 20,
        trajectory_skill_window_rounds: int = 3,
        ignore_multiple_calls: bool = True,
        max_prompt_length: int | None = None,
        max_output_length: int = 400000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.generator_prompt_template = read_file(generator_prompt_file_path.replace("/", os.sep)).lstrip()
        self.trained_playbook_file_path = trained_playbook_file_path
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
        self.ignore_multiple_calls = ignore_multiple_calls
        self.retrieve_enabled = retrieve_enabled
        self.retrieve_mode = retrieve_mode
        self.retrieval_max_skills_per_call = retrieval_max_skills_per_call
        self.prefix_max_skills = prefix_max_skills
        self.trajectory_skill_window_rounds = trajectory_skill_window_rounds
        self.partial_code_regex = r".*```python\n(.*)"
        self.full_code_regex = r"```python\n(.*?)```"

        if self.retrieve_mode not in {"prefix_subset", "trajectory_skill_window"}:
            raise ValueError(
                "retrieve_mode must be either 'prefix_subset' or 'trajectory_skill_window'."
            )
        if self.retrieve_enabled and not retrieval_prompt_file_path:
            raise ValueError("retrieval_prompt_file_path is required when retrieve_enabled=True.")

        if os.path.exists(trained_playbook_file_path):
            self.full_playbook = read_file(trained_playbook_file_path.replace("/", os.sep))
            self.playbook = self.full_playbook
        else:
            raise FileNotFoundError(f"playbook file not found at {trained_playbook_file_path}")

        self.playbook_bullet_lookup = build_playbook_bullet_lookup(self.full_playbook)
        self.retrieval_prompt_template = None
        if retrieval_prompt_file_path:
            self.retrieval_prompt_template = read_file(
                retrieval_prompt_file_path.replace("/", os.sep)
            ).lstrip()

        retrieval_model_config = retrieval_model_config or copy.deepcopy(self.generator_model_config)
        self.retrieval_model = (
            LiteLLMGenerator(**retrieval_model_config) if self.retrieve_enabled else None
        )

        self.generator_playbook = self.full_playbook
        self.last_generated_code = ""
        self.last_assistant_content = ""
        self.skill_window_rounds: list[dict[str, Any]] = []

    def initialize(self, world: AppWorld):
        super().initialize(world)
        self.generator_playbook = self.full_playbook
        self.last_generated_code = ""
        self.last_assistant_content = ""
        self.skill_window_rounds = []

        if self.log_lm_calls and self.retrieval_model is not None:
            retrieval_lm_calls_path = os.path.join(
                world.output_logs_directory, "retrieval_lm_calls.jsonl"
            )
            self.retrieval_model.log_calls_to(file_path=retrieval_lm_calls_path)

        if self.retrieve_enabled and self.retrieve_mode == "prefix_subset":
            retrieval_result = self.retrieve_playbook_subset(
                world=world,
                max_skills=self.prefix_max_skills,
                stage="prefix_init",
            )
            self.generator_playbook = retrieval_result["formatted_bullets"]
            self._write_debug_file(
                "retrieved_prefix_playbook.txt",
                self.generator_playbook,
            )

        if self.retrieve_enabled and self.retrieve_mode == "trajectory_skill_window":
            self._save_active_skill_window_snapshot()
        self.messages = self.render_generator_prompt(world)
        self.num_instruction_messages = len(self.messages)

    def render_generator_prompt(self, world: AppWorld) -> list[dict[str, str]]:
        template = Template(self.generator_prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
            indent=1,
        )
        template_params = {
            "input_str": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "relevant_apis": str(world.task.ground_truth.required_apis),
            "playbook": self.generator_playbook,
        }
        output_str = template.render(template_params)
        output_str = self.truncate_input(output_str) + "\n\n"
        return self.text_to_messages(output_str)

    def next_execution_inputs_and_cost(
        self, last_execution_outputs: list[ExecutionIO], world_gt_code: str = None
    ) -> tuple[ExecutionIO, float, str | None]:
        if last_execution_outputs:
            assert (
                len(last_execution_outputs) == 1
            ), "React expects exactly one last_execution_output."
            last_execution_output_content = last_execution_outputs[0].content
            if self.retrieve_enabled and self.retrieve_mode == "trajectory_skill_window":
                self.retrieve_playbook_subset(
                    world=self.world,
                    max_skills=self.retrieval_max_skills_per_call,
                    stage="trajectory_step",
                    last_execution_output_content=last_execution_output_content,
                )
            potential_new_line = ""
            last_execution_output_content = (
                "Output:\n```\n"
                + self.truncate_output(last_execution_output_content)
                + potential_new_line
                + "```\n\n"
            )
            self.messages.append({"role": "user", "content": last_execution_output_content})

        output = self.language_model.generate(messages=self.build_generator_messages())
        code, fixed_output_content = self.extract_code_and_fix_content(output["content"])
        self.last_generated_code = code
        self.last_assistant_content = fixed_output_content
        self.messages.append({"role": "assistant", "content": fixed_output_content + "\n\n"})
        self.logger.show_message(
            role="agent", message=fixed_output_content, step_number=self.step_number
        )
        return [ExecutionIO(content=code)], output["cost"], None

    def retrieve_playbook_subset(
        self,
        world: AppWorld,
        max_skills: int,
        stage: str,
        last_execution_output_content: str = "",
    ) -> dict[str, Any]:
        if not self.retrieval_model or not self.retrieval_prompt_template:
            raise ValueError("Retrieval model and prompt must be configured for retrieval.")

        template = Template(self.retrieval_prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in world.task.app_descriptions.items()],
            indent=1,
        )
        active_skill_ids = self.get_active_skill_window_ids()
        rendered_prompt = template.render(
            input_str=world.task.instruction,
            main_user=world.task.supervisor,
            app_descriptions=app_descriptions,
            relevant_apis=str(world.task.ground_truth.required_apis),
            playbook=self.full_playbook,
            max_skills=max_skills,
            recent_interaction=self.last_assistant_content or "(No previous model output yet)",
            recent_output=last_execution_output_content or "(No previous environment output yet)",
            active_skill_ids=json.dumps(active_skill_ids),
            active_skill_window=self.build_active_skill_window_message(include_wrapper=False)
            or "(No active retrieved skills yet)",
        )
        retrieval_messages = self.text_to_messages(rendered_prompt)
        retrieval_output = self.retrieval_model.generate(messages=retrieval_messages)
        normalized = self.normalize_retrieval_response(
            raw_response=retrieval_output.get("content") or "",
            max_skills=max_skills,
        )
        formatted_bullets = extract_playbook_bullets(
            self.full_playbook,
            normalized["selected_bullet_ids"],
            include_section_headers=True,
            empty_message="(No retrieved playbook bullets)",
        )

        event = {
            "stage": stage,
            "step_number": self.step_number,
            "selected_bullet_ids": normalized["selected_bullet_ids"],
            "retrieval_reasoning": normalized["reasoning"],
            "raw_response": retrieval_output.get("content") or "",
            "recent_interaction": self.last_assistant_content,
            "recent_output": self.truncate_output(last_execution_output_content),
            "active_skill_ids_before": active_skill_ids,
            "formatted_bullets": formatted_bullets,
        }

        if stage == "trajectory_step" and normalized["selected_bullet_ids"]:
            self.append_skill_window_round(
                selected_bullet_ids=normalized["selected_bullet_ids"],
                retrieval_reasoning=normalized["reasoning"],
            )
            event["active_skill_ids_after"] = self.get_active_skill_window_ids()
            self._save_active_skill_window_snapshot()
        elif stage == "prefix_init":
            event["active_skill_ids_after"] = normalized["selected_bullet_ids"]

        self.log_retrieval_event(event)
        return {
            "selected_bullet_ids": normalized["selected_bullet_ids"],
            "reasoning": normalized["reasoning"],
            "formatted_bullets": formatted_bullets,
        }

    def normalize_retrieval_response(self, raw_response: str, max_skills: int) -> dict[str, Any]:
        parsed_response = extract_json_from_text(raw_response) or {}
        if not isinstance(parsed_response, dict):
            parsed_response = {}

        candidate_ids = []
        for key in (
            "selected_bullet_ids",
            "bullet_ids",
            "selected_ids",
            "retrieved_bullet_ids",
        ):
            value = parsed_response.get(key)
            if isinstance(value, list):
                candidate_ids = value
                break
            if isinstance(value, str):
                candidate_ids = [value]
                break

        if not candidate_ids and isinstance(parsed_response.get("selected_bullets"), list):
            for bullet in parsed_response["selected_bullets"]:
                if isinstance(bullet, str):
                    candidate_ids.append(bullet)
                elif isinstance(bullet, dict):
                    bullet_id = bullet.get("id") or bullet.get("bullet_id")
                    if bullet_id:
                        candidate_ids.append(bullet_id)

        selected_bullet_ids = []
        for bullet_id in candidate_ids:
            normalized_id = str(bullet_id).strip()
            if normalized_id in self.playbook_bullet_lookup and normalized_id not in selected_bullet_ids:
                selected_bullet_ids.append(normalized_id)
            if len(selected_bullet_ids) >= max_skills:
                break

        reasoning = parsed_response.get("reasoning", "")
        if reasoning is None:
            reasoning = ""

        return {
            "selected_bullet_ids": selected_bullet_ids,
            "reasoning": str(reasoning).strip(),
        }

    def append_skill_window_round(
        self,
        selected_bullet_ids: list[str],
        retrieval_reasoning: str,
    ) -> None:
        self.skill_window_rounds.append(
            {
                "step_number": self.step_number,
                "selected_bullet_ids": list(selected_bullet_ids),
                "retrieval_reasoning": retrieval_reasoning,
            }
        )
        if len(self.skill_window_rounds) > self.trajectory_skill_window_rounds:
            self.skill_window_rounds = self.skill_window_rounds[
                -self.trajectory_skill_window_rounds :
            ]

    def get_active_skill_window_ids(self) -> list[str]:
        active_ids = []
        for round_entry in self.skill_window_rounds:
            for bullet_id in round_entry["selected_bullet_ids"]:
                if bullet_id not in active_ids:
                    active_ids.append(bullet_id)
        return active_ids

    def build_active_skill_window_message(self, include_wrapper: bool = True) -> str:
        if not self.skill_window_rounds:
            return ""

        lines = []
        if include_wrapper:
            lines.extend(
                [
                    "Retrieved Skill Guidance:",
                    "These skills were retrieved from the playbook using recent trajectory evidence.",
                    "Apply them in the next steps unless the environment clearly shows they are irrelevant.",
                    "",
                ]
            )

        seen_ids = set()
        for round_entry in reversed(self.skill_window_rounds):
            round_lines = []
            for bullet_id in round_entry["selected_bullet_ids"]:
                if bullet_id in seen_ids:
                    continue
                bullet = self.playbook_bullet_lookup.get(bullet_id)
                if bullet is None:
                    continue
                seen_ids.add(bullet_id)
                round_lines.append(f"[{bullet_id}] {bullet['content']}")
            if round_lines:
                lines.append(f"Recent retrieval from step {round_entry['step_number']}:")
                lines.extend(round_lines)
                lines.append("")

        if not seen_ids:
            return ""

        message = "\n".join(lines).rstrip()
        if include_wrapper:
            message += "\n\n"
        return message

    def build_generator_messages(self) -> list[dict[str, str]]:
        protected_suffix_messages = []
        if self.retrieve_enabled and self.retrieve_mode == "trajectory_skill_window":
            active_skill_window = self.build_active_skill_window_message()
            if active_skill_window:
                protected_suffix_messages.append({"role": "user", "content": active_skill_window})
        return self.build_trimmed_messages(protected_suffix_messages=protected_suffix_messages)

    def build_trimmed_messages(
        self,
        protected_suffix_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        protected_suffix_messages = copy.deepcopy(protected_suffix_messages or [])
        messages = copy.deepcopy(self.messages)
        pre_messages = messages[: self.num_instruction_messages - 1]
        post_messages = messages[self.num_instruction_messages - 1 :]
        output_str = self.messages_to_text(post_messages + protected_suffix_messages)
        remove_prefix = output_str[: output_str.index("Task: ") + 6]
        output_str = output_str.removeprefix(remove_prefix)
        observation_index = 0

        while len(output_str) > self.max_output_length:
            found_block = False
            if observation_index < len(post_messages) - 5:
                for message_index, message in enumerate(post_messages[observation_index:]):
                    if message["role"] == "user" and message["content"].startswith("Output:"):
                        message["content"] = "Output:\n```\n[NOT SHOWN FOR BREVITY]```\n\n"
                        found_block = True
                        observation_index += message_index + 1
                        break
                if not found_block:
                    observation_index = len(post_messages)

            if not found_block and len(post_messages):
                first_post_message = copy.deepcopy(post_messages[0])
                if not first_post_message["content"].endswith("[TRIMMED HISTORY]\n\n"):
                    first_post_message["content"] += "[TRIMMED HISTORY]\n\n"
                post_messages = [first_post_message] + post_messages[2:]
                found_block = True

            if not found_block:
                raise ValueError(f"No blocks found to be removed!\n{post_messages}")

            output_str = self.messages_to_text(post_messages + protected_suffix_messages)
            output_str = output_str.removeprefix(remove_prefix)

        return pre_messages + post_messages + protected_suffix_messages

    def log_retrieval_event(self, event: dict[str, Any]) -> None:
        file_path = os.path.join(self.world.output_logs_directory, "retrieval_events.jsonl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_active_skill_window_snapshot(self) -> None:
        content = self.build_active_skill_window_message(include_wrapper=True)
        if not content:
            content = "Retrieved Skill Guidance:\n(No active retrieved skills yet)\n"
        self._write_debug_file("active_skill_window.md", content)

    def _write_debug_file(self, file_name: str, content: str) -> None:
        file_path = os.path.join(self.world.output_logs_directory, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

    def extract_code_and_fix_content(self, text: str) -> tuple[str, str]:
        if text is None:
            return "", ""
        original_text = text
        output_code = ""
        match_end = 0
        for re_match in re.finditer(self.full_code_regex, original_text, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_calls:
                text = original_text[: re_match.end()]
                return code, text
            output_code += code + "\n"
            match_end = re_match.end()
        partial_match = re.match(
            self.partial_code_regex, original_text[match_end:], flags=re.DOTALL
        )
        if partial_match:
            output_code += partial_match.group(1).strip()
            if not text.endswith("\n"):
                text = text + "\n"
            text = text + "```"
        if len(output_code) == 0:
            return "", text
        return output_code, text

    def truncate_input(self, input_str: str) -> str:
        if self.max_prompt_length is None:
            return input_str
        max_prompt_length = self.max_prompt_length
        goal_index = input_str.rfind("Task:")
        if goal_index == -1:
            raise ValueError(f"No goal found in input string:\n{input_str}")
        next_new_line_index = input_str.find("\n", goal_index) + 1
        init_prompt = input_str[:next_new_line_index]
        prompt = input_str[next_new_line_index:]
        if len(init_prompt) > max_prompt_length:
            raise ValueError("Input prompt longer than max allowed length")
        if len(prompt) > max_prompt_length - len(init_prompt):
            new_prompt = prompt[-(max_prompt_length - len(init_prompt)) :]
            cmd_index = new_prompt.find("ASSISTANT:") if "ASSISTANT:" in new_prompt else 0
            prompt = "\n[TRIMMED HISTORY]\n\n" + new_prompt[cmd_index:]
        return init_prompt + prompt

    def truncate_output(self, execution_output_content: str) -> str:
        if len(execution_output_content) > 20000:
            execution_output_content = (
                execution_output_content[:20000] + "\n[REST NOT SHOWN FOR BREVITY]"
            )
        return execution_output_content

    def text_to_messages(self, input_str: str) -> list[dict]:
        messages_json = []
        last_start = 0
        for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", input_str, flags=re.IGNORECASE):
            last_end = match.span()[0]
            if len(messages_json) == 0:
                if last_end != 0:
                    raise ValueError(
                        f"Start of the prompt has no assigned role: {input_str[:last_end]}"
                    )
            else:
                messages_json[-1]["content"] = input_str[last_start:last_end]
            role = match.group(1).lower()
            messages_json.append({"role": role, "content": None})
            last_start = match.span()[1]
        messages_json[-1]["content"] = input_str[last_start:]
        return messages_json

    def messages_to_text(self, messages: list[dict]) -> str:
        output_str = ""
        for message in messages:
            role = message["role"]
            if role == "system":
                output_str += "SYSTEM:\n" + message["content"]
            elif role == "assistant":
                output_str += "ASSISTANT:\n" + message["content"]
            elif role == "user":
                output_str += "USER:\n" + message["content"]
            else:
                raise ValueError(f"Unknown message role {role} in: {message}")
        return output_str

    @property
    def trimmed_messages(self) -> list[dict]:
        return self.build_trimmed_messages()

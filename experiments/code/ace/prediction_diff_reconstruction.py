import os
import re
from typing import Any

from appworld import AppWorld
from appworld.common.utils import yield_jsonl


PREDICTION_TRIGGER_TEXT = "Before writing your next code block, predict what the environment would return"


def normalize_code(code: str) -> str:
    lines = [line.rstrip() for line in (code or "").strip().splitlines()]
    return "\n".join(lines).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def is_complete_task_action(code: str) -> bool:
    stripped_lines = [line.strip() for line in (code or "").splitlines() if line.strip()]
    if not stripped_lines:
        return False
    return bool(re.match(r"^apis\.supervisor\.complete_task\s*\(", stripped_lines[-1]))


def filter_predicted_entries_for_classification(
    predicted_entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    filtered_entries = []
    for predicted_entry in predicted_entries:
        current_code = predicted_entry.get("input") or ""
        current_predicted_output = predicted_entry.get("output") or ""
        if not current_predicted_output.strip() and is_complete_task_action(current_code):
            continue
        filtered_entries.append(predicted_entry)
    return filtered_entries


def count_effective_predicted_entries(predicted_entries: list[dict[str, str]]) -> int:
    return len(filter_predicted_entries_for_classification(predicted_entries))


def is_prediction_call(
    lm_call: dict[str, Any],
    prediction_trigger_text: str = PREDICTION_TRIGGER_TEXT,
) -> bool:
    messages = lm_call["input"]["messages"]
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if prediction_trigger_text in content:
            return True
    return False


def extract_output_content(lm_call: dict[str, Any]) -> str:
    message = lm_call.get("output", {}).get("choices", [{}])[0].get("message", {})
    return str(message.get("content") or "")


def extract_code_and_reasoning(text: str) -> tuple[str, str]:
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


def clean_predicted_output(text: str) -> str:
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


def extract_reconstructed_steps(
    lm_calls: list[dict[str, Any]],
    prediction_trigger_text: str = PREDICTION_TRIGGER_TEXT,
) -> list[dict[str, Any]]:
    steps = []
    total_calls = len(lm_calls)
    call_index = 0
    while call_index < total_calls:
        lm_call = lm_calls[call_index]
        if is_prediction_call(lm_call, prediction_trigger_text=prediction_trigger_text):
            call_index += 1
            continue

        output_content = extract_output_content(lm_call)
        code, reasoning = extract_code_and_reasoning(output_content)
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
            if is_prediction_call(next_call, prediction_trigger_text=prediction_trigger_text):
                step["prediction_raw"] = clean_predicted_output(extract_output_content(next_call))
                step["prediction_lm_call_index"] = next_index
                break
            next_output_content = extract_output_content(next_call)
            next_code, _ = extract_code_and_reasoning(next_output_content)
            if next_code.strip():
                break
            next_index += 1

        steps.append(step)
        call_index += 1
    return steps


def match_step_for_code(
    current_code: str,
    reconstructed_steps: list[dict[str, Any]],
    step_cursor: int,
    lookahead: int = 4,
) -> tuple[dict[str, Any] | None, int, dict[str, Any]]:
    normalized_current = normalize_code(current_code)
    best_match_index = None
    upper = min(len(reconstructed_steps), step_cursor + lookahead)

    for candidate_index in range(step_cursor, upper):
        candidate_code = normalize_code(reconstructed_steps[candidate_index].get("code") or "")
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
                "step_code_match": normalize_code(matched.get("code") or "") == normalized_current,
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


def build_aligned_interactions(
    task_id: str,
    environment_entries: list[dict[str, str]],
    predicted_entries: list[dict[str, str]],
    reconstructed_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    interactions: list[dict[str, Any]] = []
    step_cursor = 0
    effective_predicted_entries = filter_predicted_entries_for_classification(predicted_entries)

    if len(environment_entries) < len(effective_predicted_entries):
        raise ValueError(
            "environment_entries is shorter than predicted_entries after filtering "
            f"for task={task_id}: env={len(environment_entries)} "
            f"pred={len(effective_predicted_entries)}"
        )

    trailing_environment_entries = environment_entries[len(effective_predicted_entries) :]
    if trailing_environment_entries and not all(
        is_complete_task_action(entry.get("input") or "")
        for entry in trailing_environment_entries
    ):
        raise ValueError(
            "Found non-complete_task trailing environment interactions after prediction-driven "
            f"alignment for task={task_id}"
        )

    for index, predicted_entry in enumerate(effective_predicted_entries):
        environment_entry = environment_entries[index]
        current_code = predicted_entry.get("input") or ""
        current_actual_output = environment_entry.get("output") or ""
        current_predicted_clipped = predicted_entry.get("output") or ""
        if normalize_code(environment_entry.get("input") or "") != normalize_code(current_code):
            raise ValueError(
                "Predicted/environment code mismatch during prediction-driven alignment for "
                f"task={task_id}, interaction_index={index + 1}"
            )

        matched_step, step_cursor, alignment = match_step_for_code(
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
                    "actual_observation": past_entry.get("output") or "",
                    "actual_output": past_entry.get("output") or "",
                }
            )

        interactions.append(
            {
                "task_id": task_id,
                "interaction_index": index + 1,
                "current_reasoning": current_reasoning,
                "current_code": current_code,
                "predicted_observation": predicted_output_raw,
                "actual_observation": current_actual_output,
                "predicted_output": predicted_output_raw,
                "actual_output": current_actual_output,
                "history": history,
                "prediction_source": prediction_source,
                "alignment": alignment,
            }
        )
    return interactions


def reconstruct_interactions_from_logs(
    task_id: str,
    source_logs_dir: str,
    prediction_trigger_text: str = PREDICTION_TRIGGER_TEXT,
) -> list[dict[str, Any]]:
    environment_log_path = os.path.join(source_logs_dir, "environment_io.md")
    predicted_log_path = os.path.join(source_logs_dir, "predicted_environment_io.md")
    lm_calls_path = os.path.join(source_logs_dir, "lm_calls.jsonl")

    environment_entries = AppWorld.parse_environment_io_log(file_path=environment_log_path)
    predicted_entries = AppWorld.parse_environment_io_log(file_path=predicted_log_path)
    lm_calls = list(yield_jsonl(lm_calls_path))
    reconstructed_steps = extract_reconstructed_steps(
        lm_calls,
        prediction_trigger_text=prediction_trigger_text,
    )
    return build_aligned_interactions(
        task_id=task_id,
        environment_entries=environment_entries,
        predicted_entries=predicted_entries,
        reconstructed_steps=reconstructed_steps,
    )

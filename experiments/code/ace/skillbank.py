import copy
from typing import Any


PRIMARY_BOARDS = (
    "docs_lookup",
    "auth",
    "read_fetch",
    "local_reasoning",
)

DIFF_CATEGORIES = (
    "safe_abstraction",
    "schema_or_name_mismatch",
    "missing_decisive_information",
    "value_or_state_mismatch",
    "wrong_action_or_failure_mode",
)


def build_empty_skillbank() -> dict[str, Any]:
    return {
        "version": "v1",
        "skill_id_counter": 1,
        "buckets": {
            primary_board: {diff_category: [] for diff_category in DIFF_CATEGORIES}
            for primary_board in PRIMARY_BOARDS
        },
    }


def ensure_skillbank_shape(skillbank: dict[str, Any] | None) -> dict[str, Any]:
    normalized = copy.deepcopy(skillbank) if isinstance(skillbank, dict) else build_empty_skillbank()
    if not isinstance(normalized.get("version"), str):
        normalized["version"] = "v1"
    if not isinstance(normalized.get("skill_id_counter"), int):
        normalized["skill_id_counter"] = 1

    buckets = normalized.get("buckets")
    if not isinstance(buckets, dict):
        normalized["buckets"] = build_empty_skillbank()["buckets"]
        return normalized

    for primary_board in PRIMARY_BOARDS:
        board_bucket = buckets.get(primary_board)
        if not isinstance(board_bucket, dict):
            buckets[primary_board] = {diff_category: [] for diff_category in DIFF_CATEGORIES}
            continue
        for diff_category in list(board_bucket):
            if diff_category not in DIFF_CATEGORIES:
                board_bucket.pop(diff_category)
        for diff_category in DIFF_CATEGORIES:
            if not isinstance(board_bucket.get(diff_category), list):
                board_bucket[diff_category] = []
    for primary_board in list(buckets):
        if primary_board not in PRIMARY_BOARDS:
            buckets.pop(primary_board)
    return normalized


def get_bucket(
    skillbank: dict[str, Any],
    primary_board: str,
    diff_category: str,
) -> list[dict[str, Any]]:
    return skillbank["buckets"][primary_board][diff_category]


def count_skills(skillbank: dict[str, Any]) -> int:
    total = 0
    for primary_board in PRIMARY_BOARDS:
        for diff_category in DIFF_CATEGORIES:
            total += len(get_bucket(skillbank, primary_board, diff_category))
    return total


def next_skill_id(skillbank: dict[str, Any]) -> str:
    counter = int(skillbank["skill_id_counter"])
    skillbank["skill_id_counter"] = counter + 1
    return f"sk-{counter:05d}"


def apply_skill_operations(
    skillbank: dict[str, Any],
    operations: list[dict[str, Any]],
    primary_board: str,
    diff_category: str,
    task_id: str,
) -> list[dict[str, Any]]:
    bucket = get_bucket(skillbank, primary_board, diff_category)
    applied_operations: list[dict[str, Any]] = []

    for operation in operations:
        operation_type = operation["type"]
        if operation_type == "ADD":
            skill = operation["skill"]
            skill_id = next_skill_id(skillbank)
            added_skill = {
                "skill_id": skill_id,
                "content": skill["content"],
                "note": skill.get("note", ""),
                "source": {
                    "task_id": task_id,
                    "primary_board": primary_board,
                    "diff_category": diff_category,
                },
            }
            bucket.append(added_skill)
            applied_operations.append(
                {
                    "type": "ADD",
                    "skill_id": skill_id,
                    "skill": added_skill,
                }
            )
            continue

        if operation_type == "MODIFY":
            target_skill_id = operation["target_skill_id"]
            updated_skill = operation["updated_skill"]
            for bucket_index, existing_skill in enumerate(bucket):
                if existing_skill.get("skill_id") != target_skill_id:
                    continue
                bucket[bucket_index] = {
                    **existing_skill,
                    "content": updated_skill["content"],
                    "note": updated_skill.get("note", existing_skill.get("note", "")),
                }
                applied_operations.append(
                    {
                        "type": "MODIFY",
                        "skill_id": target_skill_id,
                        "skill": bucket[bucket_index],
                    }
                )
                break

    return applied_operations

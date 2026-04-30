import argparse
import os

from appworld.common.utils import read_json
from appworld_experiments.code.ace.skillbank import (
    DIFF_CATEGORIES,
    PRIMARY_BOARDS,
    count_skills,
    ensure_skillbank_shape,
    get_bucket,
)


SECTION_HEADER_MAP = {
    "docs_lookup": "DOCS LOOKUP",
    "auth": "AUTH",
    "read_fetch": "READ FETCH",
    "local_reasoning": "LOCAL REASONING",
}


def build_playbook_text(skillbank: dict) -> str:
    normalized_skillbank = ensure_skillbank_shape(skillbank)
    lines: list[str] = []

    for primary_board in PRIMARY_BOARDS:
        section_skills = []
        for diff_category in DIFF_CATEGORIES:
            section_skills.extend(get_bucket(normalized_skillbank, primary_board, diff_category))
        if not section_skills:
            continue

        if lines:
            lines.append("")
        lines.append(f"## {SECTION_HEADER_MAP[primary_board]}")
        for skill in section_skills:
            skill_id = skill.get("skill_id", "").strip()
            content = skill.get("content", "").strip()
            if not skill_id or not content:
                continue
            lines.append(f"[{skill_id}] {content}")

    return "\n".join(lines).rstrip() + "\n"


def default_output_path(skillbank_file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(skillbank_file_path))[0]
    return os.path.join("experiments", "playbooks", f"{stem}_playbook.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a prediction-diff skillbank JSON into an ACE-style playbook txt."
    )
    parser.add_argument("--skillbank-file-path", required=True)
    parser.add_argument("--output-file-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skillbank = read_json(args.skillbank_file_path.replace("/", os.sep))
    playbook_text = build_playbook_text(skillbank)
    output_file_path = args.output_file_path or default_output_path(args.skillbank_file_path)
    output_file_path = output_file_path.replace("/", os.sep)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(playbook_text)

    normalized_skillbank = ensure_skillbank_shape(skillbank)
    print(f"Saved playbook to {output_file_path}")
    print(f"Total skills written: {count_skills(normalized_skillbank)}")


if __name__ == "__main__":
    main()

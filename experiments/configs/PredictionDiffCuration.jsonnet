local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";

local curator_model_config = {
    "name": "gpt-5.4",
    "provider": "OpenAI",
    "base_url": "https://api.zwlbnot.cn/v1",
    "temperature": 0,
    "seed": 100,
    "stop": ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>"],
    "logprobs": false,
    "top_logprobs": null,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "response_format": {"type": "text"},
    "retry_after_n_seconds": 10,
    "use_cache": true,
    "max_retries": 50,
};

{
    "type": "ace",
    "config": {
        "run_type": "prediction-diff-curation",
        "agent": {
            "type": "prediction_diff_curator",
            "curator_model_config": curator_model_config,
            "curator_prompt_file_path": experiment_prompts_path + "/appworld_react_curator_with_classification_prompt.txt",
            "initial_playbook_file_path": experiment_playbooks_path + "/appworld_initial_playbook.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_prediction_diff_curated_playbook.txt",
            "source_experiment_name": "ReAct_non_ACE_evaluation",
            "classification_file_name": "prediction_diff_classification.jsonl",
            "max_field_chars": 100000,
            "max_classification_chars": 200000,
            "max_history_chars": 240000,
            "log_lm_calls": true,
        },
        "dataset": "train_difficulty_3",
    }
}

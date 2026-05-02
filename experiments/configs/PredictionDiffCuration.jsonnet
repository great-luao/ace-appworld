local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_skillbanks_path = project_home_path + "/experiments/skillbanks";

local curator_model_config = {
    "name": "gpt-5.3-codex",
    "provider": "OpenAI",
    "base_url": "https://api.zwlbnot.cn/v1",
    "reasoning_effort": "medium",
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
            "curator_prompt_file_path": experiment_prompts_path + "/prediction_diff_curator_prompt.txt",
            "initial_skillbank_file_path": experiment_skillbanks_path + "/appworld_prediction_diff_initial_skillbank.json",
            "trained_skillbank_file_path": experiment_skillbanks_path + "/appworld_skillbank_train_gpt5-3-codex.json",
            "source_experiment_name": "ReAct_train_gpt_5-3-codex",
            "classification_file_name": "prediction_diff_classification.jsonl",
            "max_field_chars": 500000,
            "max_history_chars": 500000,
            "log_lm_calls": false,
        },
        "dataset": "train",
    }
}

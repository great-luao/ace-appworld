local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";

local classifier_model_config = {
    "name": "gpt-5.4",
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
        "run_type": "prediction-diff-classification",
        "agent": {
            "type": "prediction_diff_classifier",
            "classifier_model_config": classifier_model_config,
            "classifier_prompt_file_path": experiment_prompts_path + "/prediction_diff_classifier_prompt_v2.txt",
            "source_experiment_name": "ReAct_non_ACE_evaluation",
            "max_interactions_per_task": null,
            "max_field_chars": 10000,
            "max_history_chars": 100000,
            "log_lm_calls": true,
        },
        "dataset": "train_difficulty_3",
        "skip_existing_outputs": true,
    }
}

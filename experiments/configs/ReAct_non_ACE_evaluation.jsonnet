local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";

local generator_model_config = {
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
        "run_type": "non-ace-evaluation",
        "agent": {
            "type": "base_react",
            "generator_model_config": generator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": false,
                "verbose": false,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/react.txt",
            "ignore_multiple_calls": true,
            "enable_output_prediction": true,
            "output_prediction_max_tokens": 5000,
            "output_prediction_stop_tokens": ["```"],
            "output_prediction_prompt_file_path": experiment_prompts_path + "/output_prediction_injection.txt",
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "max_prompt_length": 200000,
            "max_output_length": 100000,
        },
        "dataset": "train_difficulty_1",
        "skip_existing_outputs": true,
    }
}

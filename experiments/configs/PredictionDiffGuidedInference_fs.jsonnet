local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_skillbanks_path = project_home_path + "/experiments/skillbanks";
local experiment_outputs_path = project_home_path + "/experiments/outputs";

local generator_model_config = {
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
        "run_type": "prediction-diff-inference",
        "agent": {
            "type": "prediction_diff_inference_react",
            "generator_model_config": generator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": false,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/prediction_diff_generator_prompt_fs.txt",
            "retrieval_datapoints_file_path": experiment_outputs_path + "/ReAct_train_gpt_5-4/retrieval_index/datapoints.jsonl",
            "skillbank_file_path": experiment_skillbanks_path + "/appworld_skillbank_diff3_gpt3codex.json",
            "retrieval_backend_name": "hybrid_tfidf",
            "retrieval_top_k": 8,
            "retrieval_evidence_top_n": 3,
            "retrieval_min_class_score": 0.10,
            "skill_selection_mode": "multi_bucket",
            "ignore_multiple_calls": true,
            "enable_output_prediction": true,
            "output_prediction_max_tokens": 5000,
            "output_prediction_stop_tokens": ["```"],
            "output_prediction_prompt_file_path": experiment_prompts_path + "/output_prediction_injection.txt",
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
            "max_prompt_length": 900000,
            "max_output_length": 400000,
        },
        "dataset": "test_normal_difficulty_2_24",
        "skip_existing_outputs": false,
    }
}

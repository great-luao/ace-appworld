local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_skillbanks_path = project_home_path + "/experiments/skillbanks";
local experiment_outputs_path = project_home_path + "/experiments/outputs";

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
        "run_type": "prediction-diff-inference",
        "agent": {
            "type": "prediction_diff_inference_react",
            "generator_model_config": generator_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": false,
                "verbose": false,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_with_prediction_diff_skills_prompt.txt",
            "retrieval_datapoints_file_path": experiment_outputs_path + "/ReAct_train_trajectories/retrieval_index/datapoints.jsonl",
            "skillbank_file_path": experiment_skillbanks_path + "/appworld_prediction_diff_skillbank_diff3.json",
            "retrieval_backend_name": "hybrid_tfidf",
            "retrieval_top_k": 5,
            "retrieval_evidence_top_n": 5,
            "retrieval_min_class_score": 0.15,
            "skill_max_per_injection": 5,
            "skill_selection_mode": "full_skillbank",
            "enable_rollback": false,
            "max_rollbacks_per_step": 1,
            "ignore_multiple_calls": true,
            "enable_output_prediction": true,
            "output_prediction_max_tokens": 5000,
            "output_prediction_stop_tokens": ["```"],
            "output_prediction_prompt_file_path": experiment_prompts_path + "/output_prediction_injection.txt",
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal_difficulty_3_random32",
        "skip_existing_outputs": true,
    }
}

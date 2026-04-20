local project_home_path = std.extVar("APPWORLD_PROJECT_PATH");
local experiment_prompts_path = project_home_path + "/experiments/prompts";
local experiment_playbooks_path = project_home_path + "/experiments/playbooks";
local retrieval_mode = "trajectory_skill_window";
local retrieval_enabled = true;

local generator_model_config = {
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

local retrieval_model_config = generator_model_config;

{
    "type": "ace",
    "config": {
        "run_type": "ace-evaluation",
        "agent": {
            "type": "ace_evaluation_react",
            "generator_model_config": generator_model_config,
            "retrieval_model_config": retrieval_model_config,
            "appworld_config": {
                "random_seed": 123,
            },
            "logger_config": {
                "color": true,
                "verbose": false,
            },
            "generator_prompt_file_path": experiment_prompts_path + "/appworld_react_generator_with_inline_skills_prompt.txt",
            "retrieval_prompt_file_path": experiment_prompts_path + "/agentical_retrieval_trajectory_prompt.txt",
            "trained_playbook_file_path": experiment_playbooks_path + "/appworld_prediction_diff_curated_playbook_train_full90.txt",
            "retrieve_enabled": retrieval_enabled,
            "retrieve_mode": retrieval_mode,
            "retrieval_max_skills_per_call": 5,
            "prefix_max_skills": 20,
            "trajectory_skill_window_rounds": 3,
            "ignore_multiple_calls": true,
            "max_steps": 40,
            "max_cost_overall": 1000,
            "max_cost_per_task": 10,
            "log_lm_calls": true,
        },
        "dataset": "test_normal_difficulty_3",
        "skip_existing_outputs": true,
    }
}

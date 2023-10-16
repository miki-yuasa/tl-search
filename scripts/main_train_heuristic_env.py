import os
import random
from typing import Final

import torch

from tl_search.common.io import save_json
from tl_search.envs.eval import evaluate_tuned_param_replicate_models
from tl_search.envs.typing import EnemyPolicyMode
from tl_search.envs.train import (
    train_replicated_rl_agent,
)
from tl_search.envs.typing import HeuristicEnemyEnvTrainingConfig

enemy_policy_mode: Final[EnemyPolicyMode] = "patrol"
n_envs: Final[int] = 20
total_timesteps: Final[int] = 3_000_000
num_replicates: Final[int] = 3
window: Final[int] = round(total_timesteps / 100)
tuned_param_name: Final[str | None] = "ent_coef"
tuned_param_values: Final[list[float] | None] = [0.0, 0.01, 0.05, 0.1]
curriculum_training: Final[bool] = True
force_training: Final[bool] = True
step_penalty_gamma: Final[float] = 0.00

gpu: Final[int] = 2

map_path: Final[str] = "tl_search/map/maps/board_0002_obj.txt"

suffix: str = "_curr" if curriculum_training else ""
filename_common: Final[str] = f"heuristic/{enemy_policy_mode}_enemy_ppo{suffix}"
model_save_path: Final[str] = f"out/models/{filename_common}.zip"
learning_curve_path: Final[str] = f"out/plots/reward_curve/{filename_common}.png"
animation_save_path: Final[str] = f"out/plots/animation/{filename_common}.gif"

log_path: str = f"out/data/{filename_common}.json"
stats_path: str = (
    f"out/data/heuristic/stats/stats_{enemy_policy_mode}_enemy_ppo_curr.json"
)

seeds: Final[list[int]] = [random.randint(0, 10000) for _ in range(num_replicates)]

pretrained_model_path: Final[str | None] = (
    None
    if not curriculum_training
    else "out/models/heuristic/none_enemy_ppo_ent_coef_0.0.zip"
)


device = torch.device(gpu if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

print(f"Train the agent against {enemy_policy_mode} enemy policy")

if tuned_param_name is None:
    train_replicated_rl_agent(
        num_replicates,
        enemy_policy_mode,
        map_path,
        n_envs,
        seeds,
        total_timesteps,
        model_save_path,
        learning_curve_path,
        animation_save_path,
        window,
        pretrained_model_path=pretrained_model_path,
    )
else:
    log_path = log_path.replace(".json", f"_{tuned_param_name}.json")

    for tuned_param_value in tuned_param_values:
        print(f"Training with {tuned_param_name}={tuned_param_value}")
        tuned_param_suffix: Final[str] = f"_{tuned_param_name}_{tuned_param_value}"
        tuned_param: Final[dict[str, float]] = {tuned_param_name: tuned_param_value}
        if (
            os.path.exists(
                learning_curve_path.replace(".png", tuned_param_suffix + ".png")
            )
            and not force_training
        ):
            print("Skipping training because model already exists")
        else:
            train_replicated_rl_agent(
                num_replicates,
                enemy_policy_mode,
                map_path,
                n_envs,
                seeds,
                total_timesteps,
                model_save_path.replace(".zip", tuned_param_suffix + ".zip"),
                learning_curve_path.replace(".png", tuned_param_suffix + ".png"),
                animation_save_path.replace(".gif", tuned_param_suffix + ".gif"),
                window,
                step_penalty_gamma,
                tuned_param,
                pretrained_model_path=pretrained_model_path,
            )

training_config = HeuristicEnemyEnvTrainingConfig(
    enemy_policy_mode,
    seeds,
    n_envs,
    total_timesteps,
    map_path,
    model_save_path,
    learning_curve_path,
    animation_save_path,
    window,
    tuned_param_name,
    tuned_param_values,
)

save_json(training_config.to_json(indent=4), log_path)

if tuned_param_values is not None:
    evaluate_tuned_param_replicate_models(
        tuned_param_name,
        tuned_param_values,
        num_replicates,
        enemy_policy_mode,
        model_save_path,
        map_path,
        200,
        stats_path,
        step_penalty_gamma=step_penalty_gamma,
    )

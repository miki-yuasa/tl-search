import json
from typing import Literal

import numpy as np
from stable_baselines3 import PPO
from tl_search.common.typing import ObsProp

from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.train.train import simulate_model
from tl_search.envs.typing import EnemyPolicyMode, ModelStats


def evaluate_model(
    model: PPO, env: HeuristicEnemyEnv | TLMultigrid, num_episodes: int, log_path: str
) -> ModelStats:
    capture_defeat_count: int = 0
    capture_count: int = 0
    defeat_count: int = 0
    lose_count: int = 0
    rewards: list[float] = []
    timesteps: list[int] = []
    for i in range(num_episodes):
        print(f"Episode {i + 1}/{num_episodes}")
        simulate_model(model, env)

        capture: bool = env._fixed_obj.red_flag == env._blue_agent_loc
        defeat: bool = env._is_red_agent_defeated

        if capture and defeat:
            capture_defeat_count += 1
        elif capture:
            capture_count += 1
        elif defeat:
            defeat_count += 1
        else:
            lose_count += 1

        rewards.append(env._episodic_reward)
        timesteps.append(env._step_count)

    print(f"Capture defeat count: {capture_defeat_count}")
    print(f"Capture count: {capture_count}")
    print(f"Defeat count: {defeat_count}")
    print(f"Lose count: {lose_count}")
    print(f"Average reward: {np.mean(rewards)}")
    print(f"Standard deviation of reward: {np.std(rewards)}")
    print(f"Average timesteps: {np.mean(timesteps)}")
    print(f"Standard deviation of timesteps: {np.std(timesteps)}")

    model_stats: ModelStats = {
        "capture_defeat_count": capture_defeat_count,
        "capture_count": capture_count,
        "defeat_count": defeat_count,
        "lose_count": lose_count,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "timesteps_mean": np.mean(timesteps),
        "timesteps_std": np.std(timesteps),
        "rewards": rewards,
        "timesteps": timesteps,
    }

    with open(log_path, "w") as f:
        json.dump(model_stats, f, indent=4)

    return model_stats


def evaluate_replicate_models(
    num_replicates: int,
    enemy_policy_mode: EnemyPolicyMode,
    model_path: str,
    map_path: str,
    num_episodes: int,
    log_path: str,
    step_penalty_gamma: float = 0.0,
) -> ModelStats:
    model_stats: list[ModelStats] = []

    for i in range(num_replicates):
        print(f"Replicate {i + 1}/{num_replicates}")

        env = HeuristicEnemyEnv(
            enemy_policy_mode, map_path, step_penalty_gamma=step_penalty_gamma
        )
        model = PPO.load(model_path.replace(".zip", f"_{i}.zip"))

        replicate_model_stats = evaluate_model(
            model, env, num_episodes, log_path.replace(".json", f"_{i}.json")
        )

        model_stats.append(replicate_model_stats)

    capture_defeat_count: int = 0
    capture_count: int = 0
    defeat_count: int = 0
    lose_count: int = 0
    rewards: list[float] = []
    timesteps: list[int] = []

    for replicate_model_stats in model_stats:
        capture_defeat_count += replicate_model_stats["capture_defeat_count"]
        capture_count += replicate_model_stats["capture_count"]
        defeat_count += replicate_model_stats["defeat_count"]
        lose_count += replicate_model_stats["lose_count"]
        rewards += replicate_model_stats["rewards"]
        timesteps += replicate_model_stats["timesteps"]

    all_model_stats: ModelStats = {
        "capture_defeat_count": capture_defeat_count,
        "capture_count": capture_count,
        "defeat_count": defeat_count,
        "lose_count": lose_count,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "timesteps_mean": np.mean(timesteps),
        "timesteps_std": np.std(timesteps),
        "rewards": rewards,
        "timesteps": timesteps,
    }

    with open(log_path, "w") as f:
        json.dump(all_model_stats, f, indent=4)

    return all_model_stats


def evaluate_replicate_tl_models(
    num_replicates: int,
    tl_spec: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    model_path: str,
    map_path: str,
    num_episodes: int,
    log_path: str,
) -> ModelStats:
    model_stats: list[ModelStats] = []

    for i in range(num_replicates):
        print(f"Replicate {i + 1}/{num_replicates}")

        env = TLMultigrid(
            tl_spec, obs_props, atom_prep_dict, enemy_policy_mode, map_path
        )
        try:
            model = PPO.load(model_path.replace(".zip", f"_{i}.zip"))
        except FileNotFoundError:
            model = PPO.load(model_path.replace(".zip", f"_{i}"))

        replicate_model_stats = evaluate_model(
            model, env, num_episodes, log_path.replace(".json", f"_{i}.json")
        )

        model_stats.append(replicate_model_stats)

    capture_defeat_count: int = 0
    capture_count: int = 0
    defeat_count: int = 0
    lose_count: int = 0
    rewards: list[float] = []
    timesteps: list[int] = []

    for replicate_model_stats in model_stats:
        capture_defeat_count += replicate_model_stats["capture_defeat_count"]
        capture_count += replicate_model_stats["capture_count"]
        defeat_count += replicate_model_stats["defeat_count"]
        lose_count += replicate_model_stats["lose_count"]
        rewards += replicate_model_stats["rewards"]
        timesteps += replicate_model_stats["timesteps"]

    all_model_stats: ModelStats = {
        "capture_defeat_count": capture_defeat_count,
        "capture_count": capture_count,
        "defeat_count": defeat_count,
        "lose_count": lose_count,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "timesteps_mean": np.mean(timesteps),
        "timesteps_std": np.std(timesteps),
        "rewards": rewards,
        "timesteps": timesteps,
    }

    with open(log_path, "w") as f:
        json.dump(all_model_stats, f, indent=4)

    return all_model_stats


def evaluate_tuned_param_replicate_models(
    tuned_param_name: str,
    tuned_param_values: list[float],
    num_replicates: int,
    enemy_policy_mode: EnemyPolicyMode,
    model_path: str,
    map_path: str,
    num_episodes: int,
    log_path: str,
    step_penalty_gamma: float = 0.0,
):
    tuned_param_stats: list[ModelStats] = []
    for tuned_param_value in tuned_param_values:
        print(f"Evaluating {tuned_param_name} = {tuned_param_value}")
        all_rep_stats: ModelStats = evaluate_replicate_models(
            num_replicates,
            enemy_policy_mode,
            model_path.replace(
                f".zip",
                f"_{tuned_param_name}_{tuned_param_value}.zip",
            ),
            map_path,
            num_episodes,
            log_path.replace(
                f".json",
                f"_{tuned_param_name}_{tuned_param_value}.json",
            ),
            step_penalty_gamma=step_penalty_gamma,
        )

        all_rep_stats["rewards"] = []
        all_rep_stats["timesteps"] = []

        tuned_param_stats.append(all_rep_stats)

    with open(
        log_path.replace(f".json", f"_{tuned_param_name}.json").replace("stats/", ""),
        "w",
    ) as f:
        json.dump(tuned_param_stats, f, indent=4)


def evaluate_tuned_param_replicate_tl_models(
    tuned_param_name: str,
    tuned_param_values: list[float],
    num_replicates: int,
    tl_spec: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    model_path: str,
    map_path: str,
    num_episodes: int,
    log_path: str,
):
    tuned_param_stats: list[ModelStats] = []
    for tuned_param_value in tuned_param_values:
        print(f"Evaluating {tuned_param_name} = {tuned_param_value}")
        all_rep_stats: ModelStats = evaluate_replicate_tl_models(
            num_replicates,
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            model_path.replace(
                f".zip",
                f"_{tuned_param_name}_{tuned_param_value}.zip",
            ),
            map_path,
            num_episodes,
            log_path.replace(
                f".json",
                f"_{tuned_param_name}_{tuned_param_value}.json",
            ),
        )

        all_rep_stats["rewards"] = []
        all_rep_stats["timesteps"] = []

        tuned_param_stats.append(all_rep_stats)

    with open(
        log_path.replace(f".json", f"_{tuned_param_name}.json").replace("stats/", ""),
        "w",
    ) as f:
        json.dump(tuned_param_stats, f, indent=4)

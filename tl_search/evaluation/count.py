from multiprocessing import Pool
import numpy as np

from stable_baselines3 import PPO

from tl_search.common.typing import EpisodeLengthReport
from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.train.train import simulate_model
from tl_search.envs.typing import EnemyPolicyMode


def report_episode_lengths(
    num_episodes: int,
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    num_replicates: int,
    target_model_path: str,
    num_processes: int = 100,
) -> EpisodeLengthReport:
    episode_lengths: list[int] = []
    for i in range(num_replicates):
        replicate_episode_lengths: list[int] = evaluate_episode_lengths(
            num_episodes,
            enemy_policy_mode,
            map_path,
            target_model_path.replace(".zip", f"_{i}.zip"),
            num_processes,
        )
        episode_lengths.extend(replicate_episode_lengths)

    episode_length_report: EpisodeLengthReport = get_episode_length_report(
        episode_lengths
    )

    return episode_length_report


def get_episode_length_report(episode_lengths: list[int]) -> EpisodeLengthReport:
    mean_episode_length: float = float(np.mean(episode_lengths))
    std_episode_length: float = float(np.std(episode_lengths))
    max_episode_length: int = int(np.max(episode_lengths))
    min_episode_length: int = int(np.min(episode_lengths))

    episode_length_report: EpisodeLengthReport = {
        "mean": mean_episode_length,
        "std": std_episode_length,
        "max": max_episode_length,
        "min": min_episode_length,
    }

    return episode_length_report


def evaluate_episode_lengths(
    num_episodes: int,
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    target_model_path: str,
    num_processes: int,
) -> list[int]:
    with Pool(num_processes) as pool:
        episode_lengths: list[int] = pool.starmap(
            evaluate_episode_length,
            [
                (
                    enemy_policy_mode,
                    map_path,
                    target_model_path,
                )
                for _ in range(num_episodes)
            ],
        )

    return episode_lengths


def evaluate_episode_lengths_tl(
    num_episodes: int, model: PPO, env: TLMultigrid
) -> list[int]:
    episode_lengths: list[int] = [
        evaluate_episode_length_tl(model, env) for _ in range(num_episodes)
    ]

    return episode_lengths


def evaluate_episode_length(
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    target_model_path: str,
) -> int:
    env = HeuristicEnemyEnv(enemy_policy_mode, map_path)
    model = PPO.load(target_model_path, env)
    _, episode_length = simulate_model(model, env, return_episode_length=True)

    return episode_length


def evaluate_episode_length_tl(model: PPO, env: TLMultigrid) -> int:
    _, episode_length = simulate_model(model, env, return_episode_length=True)

    return episode_length

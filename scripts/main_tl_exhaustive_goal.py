from math import ceil
import multiprocessing as mp
import random
from typing import Any, Final

from tl_search.common.typing import Exclusion, ObsProp, SpecNode
from tl_search.search.search_goal import train_exh_mp
from tl_search.search.neighbor import create_all_nodes, nodes2specs

if __name__ == "__main__":
    warm_start: bool = False
    gpu: int = 0
    num_process: int = 24

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 1_000_000
    num_replicates: Final[int] = 1
    window: Final[int] = ceil(round(total_timesteps / 100))

    gpus: tuple[int, ...] = (0, 1, 2, 3)

    predicates: tuple[str, ...] = (
        "psi_gl",
        "psi_hz",
        "psi_vs",
    )

    exclusions: list[Exclusion] = ["group"]

    suffix: str = ""
    common_dir_path: str = "search/goal"
    model_save_path: str = f"out/models/{common_dir_path}/ppo{suffix}.zip"
    learning_curve_path: str = (
        f"out/plots/reward_curve/{common_dir_path}/ppo{suffix}.png"
    )
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/kl_div/goal/kl_div_ppo{suffix}.json"

    task_name: str = "SafetyCarTLGoal1-v0"
    task_config: dict[str, Any] = {
        "agent_name": "Car",
    }
    env_config: dict[str, Any] = {
        "config": task_config,
        "task_id": task_name,
        "render_mode": "rgb_array",
        "max_episode_steps": 500,
        "width": 512,
        "height": 512,
        "camera_name": "fixedfar",
        "ignore_cost": True,
    }

    policy_kwargs: dict[str, Any] = {
        "net_arch": [512, 512],
    }

    tb_log_dir: str = f"out/logs/goal_search/exh_ppo_{gpu}/"

    ppo_config: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "buffer_size": int(1e6),
        "batch_size": 2048,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "tau": 0.05,
        "tensorboard_log": tb_log_dir,
        "policy_kwargs": policy_kwargs,
    }

    her_kwargs = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": "future",
        "copy_info_dict": True,
    }

    seeds: list[int] = [random.randint(0, 10000) for _ in range(num_replicates)]

    mp.set_start_method("spawn")

    neighbor_nodes: list[SpecNode] = create_all_nodes(predicates, exclusions)
    neighbor_specs: list[str] = list(set(nodes2specs(neighbor_nodes)))

    searching_specs: list[str] = neighbor_specs[
        gpu
        * len(neighbor_nodes)
        // len(gpus) : (gpu + 1)
        * len(neighbor_nodes)
        // len(gpus)
    ]

    gpu = gpu % 4

    train_exh_mp(
        gpu,
        num_process,
        num_replicates,
        n_envs,
        seeds,
        total_timesteps,
        model_save_path,
        learning_curve_path,
        animation_save_path,
        ppo_config,
        her_kwargs,
        window,
        data_save_path,
        searching_specs,
        env_config,
        warm_start_path=None,
    )

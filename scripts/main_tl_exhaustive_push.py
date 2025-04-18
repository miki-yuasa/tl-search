from math import ceil
import multiprocessing as mp
import random
from typing import Any, Final

from tl_search.common.typing import Exclusion, ObsProp, SpecNode
from tl_search.search.search_push import train_exh_mp
from tl_search.search.neighbor import create_all_nodes, nodes2specs
from tl_search.utils import identity

if __name__ == "__main__":
    warm_start: bool = False
    gpu: int = 3
    num_process: int = 24

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 2_000_000
    num_replicates: Final[int] = 1
    window: Final[int] = ceil(round(total_timesteps / 100))

    gpus: tuple[int, ...] = (0, 1, 2, 3)

    predicates: tuple[str, ...] = (
        "psi_blk_tar",
        "psi_obs_moved",
        "psi_blk_fallen",
    )

    exclusions: list[Exclusion] = ["group"]

    suffix: str = ""
    common_dir_path: str = "search/push"
    model_save_path: str = f"out/models/{common_dir_path}/tqc{suffix}.zip"
    learning_curve_path: str = (
        f"out/plots/reward_curve/{common_dir_path}/tqc{suffix}.png"
    )
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/kl_div/push/kl_div_tqc{suffix}.json"

    obs_props: list[ObsProp] = [
        ObsProp("d_blk_tar", ["d_blk_tar"], identity),
        ObsProp("d_obs_moved", ["d_obs_moved"], identity),
        ObsProp("d_blk_fallen", ["d_blk_fallen"], identity),
    ]

    default_distance_threshold: float = 0.05
    atom_pred_dict: dict[str, str] = {
        "psi_blk_tar": f"d_blk_tar < {default_distance_threshold}",
        "psi_obs_moved": f"d_obs_moved < {default_distance_threshold}",
        "psi_blk_fallen": f"d_blk_fallen > {default_distance_threshold}",
    }

    env_config: dict[str, Any] = {
        "render_mode": "rgb_array",
        "reward_type": "dense",
        "penalty_type": "dense",
        "dense_penalty_coef": 0.01,
        "sparse_penalty_value": 10,
        "max_episode_steps": 100,
    }

    policy_kwargs: dict[str, Any] = {
        "net_arch": [512, 512, 512],
        "n_critics": 2,
    }

    tb_log_dir: str = f"out/logs/push_search/exh_tqc_{gpu}/"

    tqc_config: dict[str, Any] = {
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
        tqc_config,
        her_kwargs,
        window,
        data_save_path,
        searching_specs,
        obs_props,
        atom_pred_dict,
        env_config,
        warm_start_path=None,
    )

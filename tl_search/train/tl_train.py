import copy, datetime, os, pickle, random
import json
from typing import overload

from numpy.typing import NDArray
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from gymnasium import spaces
from tl_search.common.io import spec2title

from tl_search.common.plotter import (
    SaveOnBestTrainingRewardCallback,
    plot_ablation,
    plot_results,
)
from tl_search.envs.tl_multigrid import TLMultigrid, create_env, make_env
from tl_search.envs.train import simulate_model
from tl_search.envs.typing import EnemyPolicyMode
from tl_search.common.typing import ObsProp


def train_tl_agent(
    env: TLMultigrid,
    n_envs: int,
    seed: int | None,
    total_timesteps: int,
    rl_model_path: str,
    learning_curve_path: str,
    device: torch.device | str,
    window: int = 10,
    tuned_params: dict[str, float] | None = None,
    rep_idx: int | None = None,
    warm_start_path: str | None = None,
) -> tuple[PPO, tuple[NDArray, NDArray]]:
    """
    Train a TL RL agent

    Parameters
    -----------
    env: TLMultigrid
        the environment to train the agent on
    n_envs: int
        the number of environments to train on
    seed: int
        the seed to use for training
    total_timesteps: int
        the number of timesteps to train for
    rl_model_path: str
        the save location of the trained model
    learning_curve_path: str
        the save location of the learning curve
    window: int
        the moving average window
    step_penalty_gamma: float
        the gamma value for the step penalty

    Returns
    -------
    model: PPO
        the trained RL model
    lc: tuple[NDArray, NDArray]
        the learning curve
    """
    log_path: str = (
        f"./tmp/log/search/{env._enemy_policy_mode}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{spec2title(env._tl_spec)}/"
        + ("" if rep_idx is None else f"{rep_idx}/")
    )

    vec_env = make_vec_env(
        create_env,
        n_envs=n_envs,
        env_kwargs={
            "tl_spec": env._tl_spec,
            "obs_props": env._obs_props,
            "atom_prep_dict": env._atom_prep_dict,
            "enemy_policy_mode": env._enemy_policy_mode,
            "map_path": env._map_path,
            "is_move_clipped": env._is_move_clipped,
            "num_max_steps": env._num_max_steps,
            "render_mode": env._render_mode,
        },
        monitor_dir=log_path,
    )
    vec_env = DummyVecEnv(
        [
            make_env(
                env._tl_spec,
                env._obs_props,
                env._atom_prep_dict,
                env._enemy_policy_mode,
                env._map_path,
                env.observation_space,
                env._is_move_clipped,
                env._num_max_steps,
                env._render_mode,
            )
            for _ in range(n_envs)
        ]
    )

    monitored_vec_env = VecMonitor(vec_env, log_path)
    eval_env = copy.deepcopy(env)

    eval_callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_path)

    print("Training model")

    print(device)

    n_steps: int = 2000

    tuned_params = {} if tuned_params is None else tuned_params

    model = (
        PPO(
            "MultiInputPolicy",
            monitored_vec_env,
            verbose=True,
            seed=seed,
            n_steps=n_steps,
            batch_size=n_steps * n_envs,
            device=device,
            **tuned_params,
        )
        if warm_start_path is None
        else PPO.load(
            warm_start_path,
            monitored_vec_env,
            verbose=True,
            seed=seed,
            n_steps=n_steps,
            batch_size=n_steps * n_envs,
            device=device,
            **tuned_params,
        )
    )

    try:
        model.learn(total_timesteps)  # , callback=eval_callback)
    except:
        model = PPO(
            "MultiInputPolicy",
            monitored_vec_env,
            verbose=True,
            seed=seed,
            n_steps=n_steps,
            batch_size=n_steps * n_envs,
            device=device,
            **tuned_params,
        )
        try:
            model.learn(total_timesteps)
        except:
            raise Exception("Failed to train model")

    model.save(rl_model_path)
    try:
        lc: tuple[NDArray, NDArray] = plot_results(
            log_path, learning_curve_path, total_timesteps, window
        )
    except:
        print(env._tl_spec)
        print(log_path)
        lc = None

    return model, lc


@overload
def train_replicate_tl_agent(
    num_replicates: int,
    n_envs: int,
    env: TLMultigrid,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    tuned_param: dict[str, float] | None = None,
    warm_start_path: str | None = None,
    force_training: bool = False,
    no_returns: bool = False,
) -> list[PPO]:
    ...


@overload
def train_replicate_tl_agent(
    num_replicates: int,
    n_envs: int,
    env: TLMultigrid,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    tuned_param: dict[str, float] | None = None,
    warm_start_path: str | None = None,
    force_training: bool = False,
    no_returns: bool = True,
) -> None:
    ...


def train_replicate_tl_agent(
    num_replicates: int,
    n_envs: int,
    env: TLMultigrid,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    tuned_param: dict[str, float] | None = None,
    warm_start_path: str | None = None,
    force_training: bool = False,
    no_returns: bool = False,
) -> list[PPO] | None:
    lcs: list[tuple[NDArray, NDArray]] = []

    models: list[PPO] = []

    for i in range(num_replicates):
        print(f"Replicate {i + 1}/{num_replicates}")
        print(f"Model path: {model_save_path.replace('.zip', f'_{i}.zip')}")
        if (
            os.path.exists(model_save_path.replace(".zip", f"_{i}.zip"))
            and not force_training
        ):
            print("Model already exists, skipping training")
            model = PPO.load(
                model_save_path.replace(".zip", f"_{i}.zip"), env, device=device
            )

        else:
            model, lc = train_tl_agent(
                env,
                n_envs,
                seeds[i],
                total_time_steps,
                model_save_path.replace(".zip", f"_{i}.zip"),
                learning_curve_path.replace(".png", f"_{i}.png"),
                device,
                window,
                tuned_param,
                i,
                warm_start_path.replace(".zip", f"_{i}.zip")
                if warm_start_path
                else None,
            )
            simulate_model(
                model,
                env,
                animation_save_path.replace(".gif", f"_{i}.gif")
                if animation_save_path is not None
                else None,
            )
            lcs.append(lc)

        models.append(model)

    try:
        if len(lcs) > 0:
            plot_ablation(lcs, learning_curve_path, total_time_steps, window)
            with open(learning_curve_path.replace(".png", ".pkl"), "wb") as f:
                pickle.dump(lcs, f)
        else:
            pass
    except:
        print("Failed to plot ablation")
        print(learning_curve_path)
        print("skipping...")

    with open(model_save_path.replace(".zip", f"_seeds.json"), "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "model": model_save_path,
                "num_replicates": num_replicates,
            },
            f,
        )

    if not no_returns:
        return models
    else:
        return

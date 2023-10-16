import copy
import pickle

from numpy.typing import NDArray

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tl_search.common.plotter import (
    SaveOnBestTrainingRewardCallback,
    plot_ablation,
    plot_results,
)
from tl_search.common.seed import torch_fix_seed

from tl_search.envs.heuristic import EnemyPolicyMode, HeuristicEnemyEnv, create_env
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.envs.typing import Path


def train_replicated_rl_agent(
    num_replicates: int,
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str,
    window: int,
    step_penalty_gamma: float = 0.0,
    tuned_param: dict[str, float] | None = None,
    pretrained_model_path: str | None = None,
):
    lcs: list[tuple[NDArray, NDArray]] = []

    for i in range(num_replicates):
        print(f"Replicate {i + 1}/{num_replicates}")

        torch_fix_seed(seeds[i])

        env = create_env(enemy_policy_mode, map_path)
        model, lc = train_rl_agent(
            env,
            n_envs,
            seeds[i],
            total_time_steps,
            model_save_path.replace(".zip", f"_{i}.zip"),
            learning_curve_path.replace(".png", f"_{i}.png"),
            window,
            step_penalty_gamma,
            tuned_param,
            pretrained_model_path.replace(".zip", f"_{i}.zip")
            if pretrained_model_path is not None
            else None,
        )
        lcs.append(lc)

        simulate_model(model, env, animation_save_path.replace(".gif", f"_{i}.gif"))

    plot_ablation(lcs, learning_curve_path, total_time_steps, window)

    with open(learning_curve_path.replace(".png", ".pkl"), "wb") as f:
        pickle.dump(lcs, f)


def train_rl_agent(
    env: HeuristicEnemyEnv,
    n_envs: int,
    seed: int | None,
    total_timesteps: int,
    rl_model_path: str,
    learning_curve_path: str,
    window: int = 10,
    step_penalty_gamma: float = 0.0,
    tuned_params: dict[str, float] | None = None,
    pretrained_model_path: str | None = None,
) -> tuple[PPO, tuple[NDArray, NDArray]]:
    """
    Train a RL agent

    Parameters
    -----------
    env: HeuristicEnemyEnv
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

    Returns
    -------
    model: PPO
        the trained RL model
    lc: tuple[NDArray, NDArray]
        the learning curve
    """

    # env.seed(seed)
    log_path: str = f"./tmp/log/{env._enemy_policy_mode}/"

    vec_env = make_vec_env(
        create_env,
        n_envs=n_envs,
        env_kwargs={
            "enemy_policy_name": env._enemy_policy_mode,
            "map_path": env._map_path,
            "is_move_clipped": env._is_move_clipped,
            "step_penalty_gamma": step_penalty_gamma,
        },
        monitor_dir=log_path,
    )
    eval_env = copy.deepcopy(env)

    eval_callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_path)

    if pretrained_model_path is not None:
        print(f"Loading pretrained model from {pretrained_model_path}")
    else:
        pass

    match tuned_params:
        case None:
            match pretrained_model_path:
                case None:
                    model = PPO("MultiInputPolicy", vec_env, verbose=True, seed=seed)
                case _:
                    model = PPO.load(
                        pretrained_model_path, env=vec_env, verbose=True, seed=seed
                    )
        case _:
            match pretrained_model_path:
                case None:
                    model = PPO(
                        "MultiInputPolicy",
                        vec_env,
                        verbose=True,
                        seed=seed,
                        **tuned_params,
                    )
                case _:
                    model = PPO.load(
                        pretrained_model_path,
                        env=vec_env,
                        verbose=True,
                        seed=seed,
                        **tuned_params,
                    )

    model.learn(total_timesteps, callback=eval_callback)
    model.save(rl_model_path)
    lc: tuple[NDArray, NDArray] = plot_results(
        log_path, learning_curve_path, total_timesteps, window
    )

    return model, lc


def simulate_model(
    model: PPO,
    env: HeuristicEnemyEnv | TLMultigrid,
    animation_path: str | None = None,
    verbose: bool = False,
    terminated_only: bool = False,
    return_episode_length: bool = False,
    seed: int | None = None,
) -> tuple[Path, Path]:
    if verbose:
        print(f"Simulating model on {env._enemy_policy_mode} environment...")
    else:
        pass

    episode_length: int = 0
    obs, _ = env.reset(seed=seed)

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_length += 1
        if terminated_only and terminated:
            break
        elif terminated or truncated:
            break
        else:
            pass

    if animation_path:
        env.render_animation(animation_path)
    else:
        pass

    trajs = env.get_agent_trajs()
    env.close()

    if return_episode_length:
        return trajs, episode_length
    else:
        return trajs

import copy, datetime, os, pickle
import json
from typing import Any
import imageio

from numpy.typing import NDArray
import torch
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TQC

from tl_search.common.io import spec2title
from tl_search.common.plotter import plot_ablation, plot_results
from tl_search.envs.tl_push import TLBlockedFetchPushEnv


def train_tl_agent(
    env: TLBlockedFetchPushEnv,
    seed: int | None,
    total_timesteps: int,
    rl_model_path: str,
    learning_curve_path: str,
    device: torch.device | str,
    tqc_kwargs: dict[str, Any],
    replay_buffer_kwargs: dict[str, Any],
    window: int = 10,
    rep_idx: int | None = None,
    warm_start_path: str | None = None,
) -> tuple[TQC, tuple[NDArray, NDArray]]:
    """
    Train a TL RL agent

    Parameters
    -----------
    env: AdversarialParkingEnv
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
    model: SAC
        the trained RL model
    lc: tuple[NDArray, NDArray]
        the learning curve
    """
    log_path: str = (
        f"./tmp/log/search/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{spec2title(env._tl_spec)}/"
        + ("" if rep_idx is None else f"{rep_idx}/")
    )

    print("Training model")

    print(device)

    training_env = copy.deepcopy(env)
    print("Training env")
    model = TQC(
        env=training_env,
        verbose=1,
        device=device,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        **tqc_kwargs,
    )

    tb_log_name: str = f"tqc_{spec2title(env._tl_spec)}" + (
        "" if rep_idx is None else f"_{rep_idx}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path="out/models/push/ckpts",
        name_prefix=rl_model_path,
    )

    try:
        model.learn(
            total_timesteps, tb_log_name=tb_log_name, callback=checkpoint_callback
        )
    except:
        model = TQC(
            env=training_env,
            verbose=1,
            device=device,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            **tqc_kwargs,
        )
        try:
            model.learn(
                total_timesteps, tb_log_name=tb_log_name, callback=checkpoint_callback
            )
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


def train_replicate_tl_agent(
    num_replicates: int,
    env: TLBlockedFetchPushEnv,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    tqc_kwargs: dict[str, Any],
    replay_buffer_kwargs: dict[str, Any],
    window: int,
    warm_start_path: str | None = None,
    force_training: bool = False,
    no_returns: bool = False,
) -> list[TQC]:
    lcs: list[tuple[NDArray, NDArray]] = []

    models: list[TQC] = []

    for i in range(num_replicates):
        print(f"Replicate {i + 1}/{num_replicates}")
        print(f"Model path: {model_save_path.replace('.zip', f'_{i}.zip')}")
        if (
            os.path.exists(model_save_path.replace(".zip", f"_{i}.zip"))
            and not force_training
        ):
            print("Model already exists, skipping training")
            model = TQC.load(
                model_save_path.replace(".zip", f"_{i}.zip"), env, device=device
            )

        else:
            model, lc = train_tl_agent(
                env,
                seeds[i],
                total_time_steps,
                model_save_path.replace(".zip", f"_{i}.zip"),
                learning_curve_path.replace(".png", f"_{i}.png"),
                device,
                tqc_kwargs,
                replay_buffer_kwargs,
                window,
                i,
                (
                    warm_start_path.replace(".zip", f"_{i}.zip")
                    if warm_start_path
                    else None
                ),
            )
            simulate_model(
                model,
                env,
                (
                    animation_save_path.replace(".gif", f"_{i}.gif")
                    if animation_save_path is not None
                    else None
                ),
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


def simulate_model(
    model: TQC,
    demo_env: TLBlockedFetchPushEnv,
    animation_save_path: str | None = None,
):
    obs, _ = demo_env.reset()

    frames = []

    while True:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = demo_env.step(action)
        frames.append(demo_env.render())
        if terminated or truncated:
            break

    demo_env.close()

    if animation_save_path is not None:
        imageio.mimsave(animation_save_path, frames, fps=15, loop=0)

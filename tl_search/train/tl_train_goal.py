import copy, datetime, os, pickle
import json
from typing import Any
import imageio

from numpy.typing import NDArray
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from tl_search.common.io import spec2title
from tl_search.common.plotter import plot_ablation, plot_results
from tl_search.envs.tl_safety_builder import CustomBuilder


def train_tl_agent(
    env: CustomBuilder,
    seed: int | None,
    total_timesteps: int,
    rl_model_path: str,
    learning_curve_path: str,
    device: torch.device | str,
    algo_kwargs: dict[str, Any],
    window: int = 10,
    rep_idx: int | None = None,
    warm_start_path: str | None = None,
    continue_from_checkpoint: bool = False,
) -> tuple[PPO, tuple[NDArray, NDArray]]:
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
        f"./tmp/log/search/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{spec2title(env.task.tl_spec)}/"
        + ("" if rep_idx is None else f"{rep_idx}/")
    )

    print("Training model")

    print(device)

    ckpt_dir: str = "out/models/search/goal/ckpts"

    if continue_from_checkpoint:
        print(
            f"Search a checkpoint in {ckpt_dir} for a file starting with {rl_model_path.split('/')[-1].replace('.zip', '')}"
        )
        ckpt_files = os.listdir(ckpt_dir)
        ckpt_files = [
            f
            for f in ckpt_files
            if f.startswith(rl_model_path.split("/")[-1].replace(".zip", ""))
            and f.endswith(".zip")
        ]
        timesteps: list[float] = [int(f.split("_")[-2]) for f in ckpt_files]
        ckpt_files = [f for _, f in sorted(zip(timesteps, ckpt_files))]
        if len(ckpt_files) > 0:
            last_ckpt = ckpt_files[-1]
            model_save_path = os.path.join(ckpt_dir, last_ckpt)
            ckpt_step: int = int(last_ckpt.split("_")[-2])
            print(f"Continuing from checkpoint: {model_save_path}")
        else:
            print("No checkpoint found, training from scratch.")
            continue_from_checkpoint = False

    training_env = copy.deepcopy(env)
    print("Training policy")
    if continue_from_checkpoint:
        print(f"Loading model from the last checkpoint {model_save_path}")
        total_timesteps = total_timesteps - ckpt_step
        model = PPO.load(model_save_path, env=training_env, device=device)
    else:
        model = PPO(
            env=training_env,
            verbose=1,
            device=device,
            **algo_kwargs,
        )

    tb_log_name: str = f"ppo{spec2title(env.task.tl_spec)}" + (
        "" if rep_idx is None else f"_{rep_idx}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=ckpt_dir,
        name_prefix=rl_model_path.split("/")[-1].replace(".zip", ""),
    )

    try:
        model.learn(
            total_timesteps,
            tb_log_name=tb_log_name,
            callback=checkpoint_callback,
            reset_num_timesteps=not continue_from_checkpoint,
        )
    except:
        if continue_from_checkpoint:
            print(f"Loading model from the last checkpoint {model_save_path}")
            model = PPO.load(model_save_path, env=training_env, device=device)
        else:
            model = PPO(
                env=training_env,
                verbose=1,
                device=device,
                **algo_kwargs,
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
        print(env.task.tl_spec)
        print(log_path)
        lc = None

    return model, lc


def train_replicate_tl_agent(
    num_replicates: int,
    env: CustomBuilder,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    algo_kwargs: dict[str, Any],
    window: int,
    warm_start_path: str | None = None,
    force_training: bool = False,
    no_returns: bool = False,
    continue_from_checkpoint: bool = False,
) -> list[PPO]:
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
                seeds[i],
                total_time_steps,
                model_save_path.replace(".zip", f"_{i}.zip"),
                learning_curve_path.replace(".png", f"_{i}.png"),
                device,
                algo_kwargs,
                window,
                i,
                (
                    warm_start_path.replace(".zip", f"_{i}.zip")
                    if warm_start_path
                    else None
                ),
                continue_from_checkpoint,
            )
            if animation_save_path is not None:
                simulate_model(
                    model,
                    env,
                    animation_save_path.replace(".gif", f"_{i}.gif"),
                )
            else:
                pass

            lcs.append(lc)

        models.append(model)

    try:
        if len(lcs) > 0:
            plot_ablation(lcs, learning_curve_path, total_time_steps, window)
            os.makedirs(
                os.path.dirname(learning_curve_path.replace(".png", ".pkl")),
                exist_ok=True,
            )
            with open(learning_curve_path.replace(".png", ".pkl"), "wb") as f:
                pickle.dump(lcs, f)
        else:
            pass
    except:
        print("Failed to plot ablation")
        print(learning_curve_path)
        print("skipping...")

    os.makedirs(
        os.path.dirname(model_save_path.replace(".zip", f"_seeds.json")), exist_ok=True
    )
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
    model: PPO,
    demo_env: CustomBuilder,
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
        os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
        imageio.mimsave(animation_save_path, frames, fps=15, loop=0)

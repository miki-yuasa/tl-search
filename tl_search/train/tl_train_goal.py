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

    tb_log_name: str = rl_model_path.split("/")[-1].replace(".zip", "")

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
            # Search the checkpoint under the total timesteps
            ckpt_found: bool = False
            for ckpt in ckpt_files:
                if int(ckpt.split("_")[-2]) <= total_timesteps:
                    last_ckpt = ckpt
                    ckpt_found = True
                else:
                    break
            if ckpt_found:
                rl_model_path = os.path.join(ckpt_dir, last_ckpt)
                ckpt_step: int = int(last_ckpt.split("_")[-2])
                print(f"Continuing from checkpoint: {rl_model_path}")
            else:
                print(
                    f"No checkpoint found under {total_timesteps}, training from scratch."
                )
                continue_from_checkpoint = False
        else:
            print("No checkpoint found, training from scratch.")
            continue_from_checkpoint = False

    training_env = copy.deepcopy(env)
    print("Training policy")
    if continue_from_checkpoint:
        print(f"Loading model from the last checkpoint {rl_model_path}")
        total_timesteps = total_timesteps - ckpt_step
        model = PPO.load(rl_model_path, env=training_env, device=device)
    else:
        model = PPO(
            env=training_env,
            verbose=1,
            device=device,
            **algo_kwargs,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=ckpt_dir,
        name_prefix=rl_model_path.split("/")[-1].replace(".zip", ""),
    )

    # Check if the env can continue more than one step
    count_env = copy.deepcopy(env)
    count_env.reset()
    step_count: int = 0
    while True:
        step_count += 1
        _, _, terminated, truncated, _ = count_env.step(count_env.action_space.sample())
        if terminated or truncated:
            break

    if step_count > 1:
        try:
            model.learn(
                total_timesteps,
                tb_log_name=tb_log_name,
                callback=checkpoint_callback,
                reset_num_timesteps=not continue_from_checkpoint,
            )
        except:
            if continue_from_checkpoint:
                print(f"Loading model from the last checkpoint {rl_model_path}")
                model = PPO.load(rl_model_path, env=training_env, device=device)
            else:
                model = PPO(
                    env=training_env,
                    verbose=1,
                    device=device,
                    **algo_kwargs,
                )
            try:
                model.learn(
                    total_timesteps,
                    tb_log_name=tb_log_name,
                    callback=checkpoint_callback,
                )
            except:
                raise Exception("Failed to train model")
    else:
        print(
            f"Skipping training for {env.task.tl_spec}, the environment can't continue more than one step"
        )

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
    num_replicates: list[str],
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
    suboptimal_ckpt_timestep: int | None = None,
) -> list[PPO]:
    lcs: list[tuple[NDArray, NDArray]] = []

    models: list[PPO] = []

    for rep_num in num_replicates:
        print(f"Replicate {rep_num} from {num_replicates}")
        print(f"Model path: {model_save_path.replace('.zip', f'_{rep_num}.zip')}")
        if suboptimal_ckpt_timestep is not None:
            print(
                f"Suboptimal checkpoint timestep: {suboptimal_ckpt_timestep} for {rep_num}"
            )
            suboptimal_model_dir: str = os.path.dirname(model_save_path) + "/ckpts"
            suboptimal_model_path: str = os.path.join(
                suboptimal_model_dir,
                f"{model_save_path.split('/')[-1].replace('.zip', '')}_{rep_num}_{suboptimal_ckpt_timestep}_steps.zip",
            )
            model = PPO.load(suboptimal_model_path, env, device=device)
        elif (
            suboptimal_ckpt_timestep is None
            and os.path.exists(model_save_path.replace(".zip", f"_{rep_num}.zip"))
            and not force_training
        ):
            print("Model already exists, skipping training")
            model = PPO.load(
                model_save_path.replace(".zip", f"_{rep_num}.zip"), env, device=device
            )

        else:
            model, lc = train_tl_agent(
                env,
                seeds[int(rep_num)],
                total_time_steps,
                model_save_path.replace(".zip", f"_{rep_num}.zip"),
                learning_curve_path.replace(".png", f"_{rep_num}.png"),
                device,
                algo_kwargs,
                window,
                int(rep_num),
                (
                    warm_start_path.replace(".zip", f"_{rep_num}.zip")
                    if warm_start_path
                    else None
                ),
                continue_from_checkpoint,
            )
            if animation_save_path is not None:
                simulate_model(
                    model,
                    env,
                    animation_save_path.replace(".gif", f"_{rep_num}.gif"),
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

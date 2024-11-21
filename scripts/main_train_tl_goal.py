import os
from typing import Any, Literal

import imageio
import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from tl_search.common.io import spec2title
from tl_search.envs.tl_safety_builder import CustomBuilder


tl_spec: str = "F(psi_gl) & G(!psi_hz & !psi_vs)"
gpu_id: int = 1
total_timesteps: int = 600_000
task_name: str = "SafetyCarTLGoal1-v0"

continue_from_checkpoint: bool = False
replay_only: bool = True

tl_title: str = spec2title(tl_spec)
net_arch = [512, 512]  # [256, 256]
replicate: str = "0"
file_title: str = (
    f"goal_ppo_{tl_title}_{net_arch[0]}_{total_timesteps/1_000_000}M_{replicate}"
)
common_dir_path: str = "search/goal"
model_save_path: str = f"out/models/{common_dir_path}/{file_title}.zip"
animation_save_path: str = f"out/plots/animation/{common_dir_path}/{file_title}.gif"
tb_log_path: str = "out/logs/tl_goal"
ckpt_dir: str = "out/models/search/goal/ckpts"

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
task_config: dict[str, Any] = {
    "agent_name": "Car",
    "tl_spec": tl_spec,
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

env = CustomBuilder(**env_config)

if continue_from_checkpoint:
    ckpt_files = os.listdir(ckpt_dir)
    ckpt_files = [
        f for f in ckpt_files if f.startswith(file_title) and f.endswith(".zip")
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

if os.path.exists(model_save_path) and not continue_from_checkpoint:
    print(f"Loading the saved model")
    model = PPO.load(model_save_path, device=device)

else:
    if continue_from_checkpoint:
        print(f"Loading model from the last checkpoint {model_save_path}")
        total_timesteps = total_timesteps - ckpt_step
        model = PPO.load(model_save_path, env=env, device=device)
    elif not replay_only:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tb_log_path,
            device=device,
            policy_kwargs=dict(net_arch=net_arch),
        )
    else:
        pass

    if not replay_only:
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=ckpt_dir,
            name_prefix=file_title,
        )

        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=file_title,
            callback=checkpoint_callback,
            reset_num_timesteps=not continue_from_checkpoint,
        )

        model.save(model_save_path)

        env.close()
    else:
        pass


demo_env = CustomBuilder(**env_config)

obs, _ = demo_env.reset()
print(obs)

frames = []

step: int = 0
ep_reward: float = 0.0
while True:
    action = model.predict(obs)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print(obs)
    print(reward)
    frames.append((255 - demo_env.render() * 255).astype(np.uint8))
    step += 1
    ep_reward += reward
    if terminated or truncated:
        print(obs)
        print("total steps:", step)
        print("total reward:", ep_reward)
        break

demo_env.close()

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=15)

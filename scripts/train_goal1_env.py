import os
from typing import Literal

import imageio
import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import safety_gymnasium
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium

from tl_search.envs.tl_safety_builder import CustomBuilder

env_type: Literal["vanilla", "terminal"] = "terminal"

gpu_id: int = 0
total_timesteps: int = 500_000

model_save_path: str = "out/models/safety_car_goal1/ppo_0"
animation_save_path: str = "out/plots/animations/safety_car_goal1/ppo_0.gif"
tb_log_path: str = "out/logs/safety_car_goal1/"

net_arch = [256, 256]

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

if env_type == "vanilla":
    env = safety_gymnasium.vector.make(
        "SafetyCarGoal1-v0", render_mode="rgb_array", max_episode_steps=100
    )
    env = SafetyGymnasium2Gymnasium(env)
else:
    env = CustomBuilder(
        config={"agent_name": "Car"},
        task_id="SafetyCarGoal1-v0",
        render_mode="rgb_array",
        max_episode_steps=500,
        width=512,
        height=512,
        camera_name="fixedfar",
    )

if not os.path.exists(model_save_path + ".zip"):

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tb_log_path,
        device=device,
        policy_kwargs=dict(net_arch=net_arch),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="out/models/safety_car_goal1/ckpts",
        name_prefix=model_save_path.split("/")[-1],
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_save_path)

    env.close()
else:
    model = PPO.load(model_save_path, device=device)

if env_type == "vanilla":
    demo_env = safety_gymnasium.vector.make(
        "SafetyCarGoal1-v0",
        render_mode="rgb_array",
        max_episode_steps=100,
        camera_name="fixedfar",
        width=512,
        height=512,
    )
    demo_env = SafetyGymnasium2Gymnasium(demo_env)
else:
    demo_env = CustomBuilder(
        config={"agent_name": "Car"},
        task_id="SafetyCarGoal1-v0",
        render_mode="rgb_array",
        max_episode_steps=500,
        width=512,
        height=512,
        camera_name="fixedfar",
    )

obs, _ = demo_env.reset()
print(obs)

frames = []

step: int = 0
while True:
    action = model.predict(obs)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print(obs)
    print(reward)
    frames.append((255 - demo_env.render() * 255).astype(np.uint8))
    step += 1
    if terminated or truncated:
        print(obs)
        print("total steps:", step)
        break

demo_env.close()

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=15)

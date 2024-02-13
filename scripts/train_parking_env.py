from typing import Literal
import numpy as np
from stable_baselines3 import PPO, SAC, HerReplayBuffer
import torch
import imageio

from tl_search.envs.parking import AdversarialParkingEnv

use_saved_model: bool = False

total_timesteps = 50_000
net_arch: list[int] = [512 for _ in range(3)]

rl_algo: Literal["ppo", "sac"] = "sac"

tb_log_path: str = "out/logs/parking_demo"

net_arch_str = "_".join(map(str, net_arch))
model_save_path: str = (
    f"out/models/parking/parking_demo_fixed_{rl_algo}_{net_arch_str}_timesteps_{total_timesteps/1_000_000}M.zip"
)
animation_save_path: str = (
    f"out/plots/animation/parking_demo_fixed_{rl_algo}_{net_arch_str}_timesteps_{total_timesteps/1_000_000}M.gif"
)
gpu_id: int = 0

her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future")


config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "reward_weights": [1, 1, 0, 0, 0.1, 0.1],
    "success_goal_reward": 0.12,
    "collision_reward": -5,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 100,
    "screen_width": 600,
    "screen_height": 300,
    "screen_center": "centering_position",
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 1,
    "add_walls": True,
    "adversarial_vehicle_spawn_config": [
        {"spawn_point": [-30, 4], "heading": 0, "speed": 5},
        {"spawn_point": [-30, -4], "heading": 0, "speed": 5},
        {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
        {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
    ],
}


device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env = AdversarialParkingEnv(config)

if not use_saved_model:
    model = (
        PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=tb_log_path,
            policy_kwargs=dict(net_arch=net_arch),
        )
        if rl_algo == "ppo"
        else SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            verbose=1,
            tensorboard_log=tb_log_path,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=1024,
            tau=0.05,
            policy_kwargs=dict(net_arch=net_arch),
        )
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_save_path)
    env.close()

else:
    model = (
        PPO.load(model_save_path, env, device=device)
        if rl_algo == "ppo"
        else SAC.load(model_save_path, env, device=device)
    )

demo_env = AdversarialParkingEnv(config)

obs, _ = demo_env.reset()
print(obs)

frames = []

while True:
    action = demo_env.action_space.sample()
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print(reward)
    frames.append(demo_env.render())
    if terminated or truncated:
        print(reward)
        break

demo_env.close()

imageio.mimsave(animation_save_path, frames, fps=15, loop=0)

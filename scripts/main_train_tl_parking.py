from typing import Literal
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
import torch
import imageio
from tl_search.common.io import spec2title

from tl_search.envs.tl_parking import TLAdversarialParkingEnv

use_saved_model: bool = True

tl_spec: str = "F(psi_ego_goal) & G(!psi_ego_adv & !psi_ego_wall)"

total_timesteps = 100_000
net_arch: list[int] = [512 for _ in range(3)]

rl_algo: Literal["sac"] = "sac"

tb_log_path: str = "out/logs/tl_parking"

net_arch_str = "_".join(map(str, net_arch))
suffix: str = (
    f"{rl_algo}_{net_arch_str}_timesteps_{total_timesteps/1_000_000}M_tl_{spec2title(tl_spec)}_scaled"
)
model_save_path: str = f"out/models/parking/parking_demo_fixed_{suffix}.zip"
animation_save_path: str = f"out/plots/animation/parking_demo_fixed_{suffix}.gif"
gpu_id: int = 0

her_kwargs = dict(
    n_sampled_goal=4, goal_selection_strategy="future", copy_info_dict=True
)


config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],  # "heading"],
        "scales": [1, 1, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "reward_weights": [1, 0.2, 0, 0, 0.1, 0.1],
    "success_goal_reward": 0.05,
    "collision_reward": -5,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 50,
    "screen_width": 600,
    "screen_height": 300,
    "screen_center": "centering_position",
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "adversarial_vehicle": True,
    "add_walls": True,
    "adversarial_vehicle_spawn_config": [
        {"spawn_point": [-30, 4], "heading": 0, "speed": 5},
        {"spawn_point": [-30, -4], "heading": 0, "speed": 5},
        {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
        {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
    ],
    "dense_reward": True,
}


device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env = TLAdversarialParkingEnv(tl_spec, config)

if not use_saved_model:
    model = SAC(
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
        device=device,
        learning_starts=1000,
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(model_save_path)
    env.close()

else:
    model = SAC.load(model_save_path, env, device=device)

demo_env = TLAdversarialParkingEnv(tl_spec, config)

obs, _ = demo_env.reset()
print(obs)

frames = []

while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print(reward)
    frames.append(demo_env.render())
    if terminated or truncated:
        print(demo_env.controlled_vehicles[0].crashed)
        print(obs)
        break

demo_env.close()

imageio.mimsave(animation_save_path, frames, fps=15, loop=0)

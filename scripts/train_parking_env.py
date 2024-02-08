import numpy as np
from stable_baselines3 import PPO
import torch
import imageio

from tl_search.envs.parking import AdversarialParkingEnv

total_timesteps = 500_000
net_arch: list[int] = [256, 256]
tb_log_path: str = "out/logs/parking_demo"

net_arch_str = "_".join(map(str, net_arch))
model_save_path: str = f"out/models/parking/parking_demo_{net_arch_str}"
animation_save_path: str = f"out/plots/animation/parking_demo_{net_arch_str}.gif"
gpu_id: int = 0


config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
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
    "vehicles_count": 4,
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

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    device=device,
    tensorboard_log=tb_log_path,
    policy_kwargs=dict(net_arch=[64, 64]),
)
model.learn(total_timesteps=total_timesteps)

model.save(model_save_path)
env.close()

demo_env = AdversarialParkingEnv(config)

obs, _ = demo_env.reset()
print(obs)

frames = []

while True:
    action = demo_env.action_space.sample()
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print(obs)
    frames.append(demo_env.render())
    if terminated or truncated:
        obs = demo_env.reset()
        print(obs)
        break

demo_env.close()

imageio.mimsave(animation_save_path, frames, fps=4, loop=0)

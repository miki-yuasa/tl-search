import os

import imageio

import torch
from stable_baselines3 import PPO
import safety_gymnasium
from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium

gpu_id: int = 1

model_save_path: str = "out/models/safety_car_goal1/ppo_0"
animation_save_path: str = "out/animations/safety_car_goal1/ppo_0.gif"
tb_log_path: str = "out/logs/safety_car_goal1/"

net_arch = [256, 256]

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env = safety_gymnasium.vector.make(
    "SafetyCarGoal1-v0", render_mode="rgb_array", max_episode_steps=100
)
env = SafetyGymnasium2Gymnasium(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=tb_log_path,
    device=device,
    policy_kwargs=dict(net_arch=net_arch),
)

model.learn(total_timesteps=100000)

model.save(model_save_path)

env.close()

demo_env = safety_gymnasium.vector.make("SafetyCarGoal1-v0", render_mode="rgb_array")
model = PPO.load(model_save_path, demo_env, device=device)

obs, _ = demo_env.reset()
print(obs)

frames = []

while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print(reward)
    frames.append(demo_env.render())
    if terminated or truncated:
        print(obs)
        break

demo_env.close()

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=15)

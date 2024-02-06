from stable_baselines3 import PPO
import torch
import imageio

from tl_search.envs.parking import AdversarialParkingEnv

total_timesteps = 100_000
tb_log_path: str = "out/logs/parking_demo"
model_save_path: str = "out/models/parking/parking_demo"
animation_save_path: str = "out/plots/animation/parking_demo.gif"
gpu_id: int = 0


device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env = AdversarialParkingEnv()

model = PPO(
    "MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=tb_log_path
)
model.learn(total_timesteps=total_timesteps)

model.save(model_save_path)
env.close()

demo_env = AdversarialParkingEnv()

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

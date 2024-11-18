import os
from typing import Any

import torch
import imageio
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import TQC

from tl_search.common.io import spec2title
from tl_search.envs.tl_push import TLBlockedFetchPushEnv

gpu_id: int = 0
total_timesteps: int = 2_000_000

tl_spec: str = "F(psi_blk_tar) & G(!psi_obs_moved & !psi_blk_fallen)"

restart_from_the_last_checkpoint: bool = False
replay_only: bool = False
replicate: int = 0

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

env_config: dict[str, Any] = {
    "tl_spec": tl_spec,
    "render_mode": "rgb_array",
    "reward_type": "dense",
    "penalty_type": "dense",
    "dense_penalty_coef": 0.01,
    "sparse_penalty_value": 10,
    "max_episode_steps": 100,
}

policy_kwargs: dict[str, Any] = {
    "net_arch": [512, 512, 512],
    "n_critics": 2,
}

tl_title: str = spec2title(tl_spec)
file_title: str = f"push_tqc_{total_timesteps/1_000_000}M_timesteps_{replicate}"
common_dir_path: str = "search/push"
model_save_path: str = f"out/models/{common_dir_path}/{file_title}.zip"
animation_save_path: str = f"out/plots/{common_dir_path}/{file_title}.gif"
tb_log_path: str = f"out/logs/tl_push"

if replay_only and restart_from_the_last_checkpoint:
    raise ValueError("Both replay_only and restart_from_the_last_checkpoint are True")
else:
    pass

if replay_only or restart_from_the_last_checkpoint:
    ckpt_dir: str = "out/models/ckpts"
    ckpt_files = os.listdir(ckpt_dir)
    ckpt_files = [
        f for f in ckpt_files if f.startswith(file_title) and f.endswith(".zip")
    ]
    timesteps: list[float] = [int(f.split("_")[-2]) for f in ckpt_files]
    ckpt_files = [f for _, f in sorted(zip(timesteps, ckpt_files))]
    if len(ckpt_files) > 0:
        last_ckpt = ckpt_files[-1]
        ckpt_step = int(last_ckpt.split("_")[-2])
        model_save_path = os.path.join(ckpt_dir, last_ckpt)
        print(f"Loading model from the last checkpoint: {model_save_path}")
    else:
        print("No checkpoint file found, starting from scratch")

tqc_config: dict[str, Any] = {
    "policy": "MultiInputPolicy",
    "buffer_size": int(1e6),
    "batch_size": 2048,
    "gamma": 0.95,
    "learning_rate": 0.001,
    "tau": 0.05,
    "tensorboard_log": tb_log_path,
    "policy_kwargs": policy_kwargs,
}
her_config: dict[str, Any] = {
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "copy_info_dict": True,
}

env = TLBlockedFetchPushEnv(**env_config)

if os.path.exists(model_save_path) and not restart_from_the_last_checkpoint:
    print(f"Loading model from the saved file: {model_save_path}")
    model = TQC.load(model_save_path, env=env, device=device)

else:
    if restart_from_the_last_checkpoint:
        print(f"Continuing training from the last checkpoint: {model_save_path}")
        total_timesteps = total_timesteps - ckpt_step
        model = TQC.load(model_save_path, env=env, device=device)
        model.learning_starts = ckpt_step + 1000

    else:
        model = TQC(
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_config,
            verbose=1,
            device=device,
            **tqc_config,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path="out/models/push/ckpts",
        name_prefix=file_title,
    )

    model.learn(
        total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=file_title,
        reset_num_timesteps=not restart_from_the_last_checkpoint,
    )

    model.save(model_save_path)

demo_env = TLBlockedFetchPushEnv(**env_config)
obs, _ = demo_env.reset()
frames = [demo_env.render()]

ep_reward: float = 0
rewards: list[float] = []
while True:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = demo_env.step(action)
    print("Observation:")
    print(obs)
    print(f"Reward:{reward}")
    ep_reward += reward
    rewards.append(reward)
    frames.append(demo_env.render())
    if terminated or truncated:
        break

print(f"Episode reward: {ep_reward}")
print(f"Rewards: {rewards}")

demo_env.close()

os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
imageio.mimsave(animation_save_path, frames, fps=30, loop=10)

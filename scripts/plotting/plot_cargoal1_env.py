from typing import Any

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from tl_search.envs.tl_safety_builder import CustomBuilder

plot_save_path: str = f"out/plots/car_goal1_sample.png"

env_config: dict[str, Any] = {
    "config": {"agent_name": "Car"},
    "task_id": "SafetyCarGoal1-v0",
    "render_mode": "rgb_array",
    "max_episode_steps": 500,
    "width": 1024,
    "height": 512,
    "camera_name": "fixednear",
}

env = CustomBuilder(**env_config)

obs, _ = env.reset()
print(obs)

img = env.render()
img = (255 - img * 255).astype(np.uint8)


ax: Axes
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(plot_save_path, dpi=600, bbox_inches="tight", pad_inches=0)

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from tl_search.envs.parking import AdversarialParkingEnv

plot_save_path: str = "out/plots/parking_sample.png"

config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],  # "heading"],
        "scales": [100, 100, 5, 5, 1, 1],
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
    "offscreen_rendering": True,
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
}


demo_env = AdversarialParkingEnv(config)

obs, _ = demo_env.reset()
image = demo_env.render()
demo_env.close()

ax: Axes
fig, ax = plt.subplots()
ax.imshow(image)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(plot_save_path, dpi=600, bbox_inches="tight", pad_inches=0)

from typing import Optional
from highway_env.envs import AbstractEnv
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.parking_env import ParkingEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
import numpy as np
import gymnasium as gym


class AdversarialParkingEnv(ParkingEnv):
    def __init__(
        self, config: dict | None = None, render_mode: str | None = "rgb_array"
    ) -> None:
        super().__init__(config, render_mode)

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
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
                "vehicles_count": 0,
                "add_walls": True,
            }
        )
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(
                self.road, [i * 20, 0], 2 * np.pi * self.np_random.uniform(), 0
            )
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)

        # Goal
        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(
            self.road, lane.position(lane.length / 2, 0), heading=lane.heading
        )
        self.road.objects.append(self.goal)

        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            lane = ("a", "b", i) if self.np_random.uniform() >= 0.5 else ("b", "c", i)
            v = Vehicle.make_on_lane(self.road, lane, 4, speed=0)
            self.road.vehicles.append(v)

        # Random vehicles
        vehicle = IDMVehicle(
            self.road, [10, 4], target_speed=10, target_lane_index=("a", "b", 1)
        )
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = AbsoluteCenterEnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == "rgb_array":
            image = self.viewer.get_image()
            return image


class AbsoluteCenterEnvViewer(EnvViewer):
    def __init__(self, env: AbstractEnv, config: dict | None = None) -> None:
        super().__init__(env, config)

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.config["screen_center"] == "centering_position":
            return self.config["centering_position"]
        elif self.observer_vehicle:
            return self.observer_vehicle.position
        elif self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

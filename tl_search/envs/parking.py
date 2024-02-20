from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

from highway_env.envs import AbstractEnv
from highway_env.envs.common.action import action_factory, ActionType
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.observation import KinematicsGoalObservation
from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces
import pandas as pd


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
                "vehicles_count": 10,
                "add_walls": True,
                "adversarial_vehicle_spawn_config": [
                    {"spawn_point": [-30, 4], "heading": 0, "speed": 5},
                    {"spawn_point": [-30, -4], "heading": 0, "speed": 5},
                    {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
                    {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
                ],
            }
        )
        return config

    def define_spaces(self) -> None:
        self.observation_type = KinematicGoalVehiclesObservation(
            self, **self.config["observation"]
        )
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()
        self.observation_type_parking = KinematicsGoalObservation(
            self, **self.config["observation"]
        )

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
        # goal_lane = self.road.network.lanes_list()[4]
        goal_lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = CustomLandmark(
            self.road,
            goal_lane.position(goal_lane.length / 2, 0),
            heading=goal_lane.heading,
        )
        self.road.objects.append(self.goal)

        # Other vehicles
        selected_lane_indexes: list[tuple[str, str, int]] = [
            ("a", "b", 9),
            ("b", "c", 11),
            ("a", "b", 10),
            ("b", "c", 0),
            ("a", "b", 6),
            ("b", "c", 8),
            ("b", "c", 2),
            ("a", "b", 0),
            ("b", "c", 10),
            ("b", "c", 1),
        ]
        for i in range(self.config["vehicles_count"]):
            while True:
                lane_id = self.np_random.choice(
                    range(0, int(len(self.road.network.lanes_list()) / 2))
                )
                lane_index = (
                    ("a", "b", lane_id)
                    if self.np_random.uniform() >= 0.5
                    else ("b", "c", lane_id)
                )

                if (
                    goal_lane != self.road.network.get_lane(lane_index)
                    and lane_index not in selected_lane_indexes
                ):
                    selected_lane_indexes.append(lane_index)
                    break

            v = Vehicle.make_on_lane(self.road, selected_lane_indexes[i], 4, speed=0)
            self.road.vehicles.append(v)

        # Adversarial vehicle
        if self.config["adversarial_vehicle"]:
            vehicle_config: dict = self.np_random.choice(
                self.config["adversarial_vehicle_spawn_config"]
            )
            vehicle = RandomVehicle(
                self.road,
                vehicle_config["spawn_point"],
                vehicle_config["heading"],
                vehicle_config["speed"],
            )
            # vehicle.color = VehicleGraphics.PURPLE
            self.road.vehicles.append(vehicle)
        else:
            pass

        # Walls
        if self.config["add_walls"]:
            width, height = 70, 42
            self.wall_width: float = width
            self.wall_height: float = height
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

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)

        achieved_goal = obs["achieved_goal"]

        obs = self.observation_type_parking.observe()
        success = self._is_success(achieved_goal, obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        # obs = obs if isinstance(obs, tuple) else (obs,)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {})
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )

        # if self.goal.hit:
        #     reward += 0.12

        return reward

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )

        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class KinematicGoalVehiclesObservation(KinematicsGoalObservation):
    def __init__(self, env: AbstractEnv, scales: List[float], **kwargs: dict) -> None:
        super().__init__(env, scales, **kwargs)
        self.vehicles_count: int = (
            env.config["vehicles_count"] + env.config["controlled_vehicles"]
        )

    def space(self) -> spaces.Space:
        obs = self.observe()
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["observation"].shape,
                    dtype=np.float64,
                ),
                "achieved_goal": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["achieved_goal"].shape,
                    dtype=np.float64,
                ),
                "desired_goal": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["desired_goal"].shape,
                    dtype=np.float64,
                ),
            }
        )

    def observe(self, **kwargs: dict) -> dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return {
                "observation": np.zeros((self.vehicles_count, len(self.features))),
                "achieved_goal": np.zeros((len(self.features),)),
                "desired_goal": np.zeros((len(self.features),)),
            }

        # Add ego-vehicle
        ego_df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[
            self.features
        ]

        # Add all other vehicles
        other_vehicles = self.env.road.vehicles[1:].copy()

        origin = self.observer_vehicle
        all_df = pd.concat(
            [
                ego_df,
                pd.DataFrame.from_records(
                    [
                        v.to_dict(origin, observe_intentions=False)
                        for v in other_vehicles
                        if v is not origin
                    ]
                ),
            ]
        )

        veh_obs: NDArray[np.float_] = all_df[self.features].to_numpy(dtype=np.float_)

        ego_obs: NDArray[np.float_] = np.ravel(ego_df)
        goal: NDArray[np.float_] = np.ravel(
            pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features]
        )
        obs: dict[str, NDArray] = {
            "observation": np.ravel(veh_obs / self.scales),
            "achieved_goal": ego_obs / self.scales,
            "desired_goal": goal / self.scales,
        }
        return obs


class RandomVehicle(Vehicle):
    ACCELERATION_RANGE: tuple[float, float] = (-5, 5.0)

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        predition_type: str = "constant_steering",
    ):
        super().__init__(road, position, heading, speed, predition_type)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action
        else:
            self.action = {
                "steering": 0,
                "acceleration": self.road.np_random.uniform(
                    low=self.ACCELERATION_RANGE[0], high=self.ACCELERATION_RANGE[1]
                ),
            }


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


class CustomLandmark(Landmark):
    """Landmarks of certain areas on the road that must be reached."""

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = super().to_dict(origin_vehicle, observe_intentions)
        d["heading"] = self.heading

        return d

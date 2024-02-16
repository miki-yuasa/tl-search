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
            # while True:
            #     lane_id = self.np_random.choice(
            #         range(0, int(len(self.road.network.lanes_list()) / 2))
            #     )
            #     lane_index = (
            #         ("a", "b", lane_id)
            #         if self.np_random.uniform() >= 0.5
            #         else ("b", "c", lane_id)
            #     )

            #     if (
            #         goal_lane != self.road.network.get_lane(lane_index)
            #         and lane_index not in selected_lane_indexes
            #     ):
            #         selected_lane_indexes.append(lane_index)
            #         break

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

    # def compute_reward(
    #     self,
    #     achieved_goal: NDArray[np.float_],
    #     desired_goal: NDArray[np.float_],
    #     info: dict,
    #     p: float = 0.5,
    # ) -> np.float_ | NDArray[np.float_]:
    #     """
    #     Proximity to the goal is rewarded

    #     We use a weighted p-norm

    #     :param achieved_goal: the goal that was achieved
    #     :param desired_goal: the goal that was desired
    #     :param dict info: any supplementary information
    #     :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
    #     :return: the corresponding reward
    #     """

    #     num_features: int = len(self.config["observation"]["features"])

    #     achieved_goal = achieved_goal.reshape([-1, num_features])
    #     desired_goal = desired_goal.reshape([-1, num_features])

    #     # Compute the reward
    #     reward: NDArray[np.float_] = np.ones((achieved_goal.shape[0], 1)) * 0

    #     # Compute the distance between the goal and the achieved goal
    #     d = np.linalg.norm(
    #         achieved_goal[:, 0:2] - desired_goal[:, 0:2], axis=1
    #     ).reshape(reward.shape)
    #     velocity_diff = np.linalg.norm(
    #         achieved_goal[:, 2:4] - desired_goal[:, 2:4], axis=1
    #     ).reshape(reward.shape)
    #     angle_diff = (achieved_goal[:, -1] - desired_goal[:, -1]).reshape(reward.shape)

    #     differences: NDArray[np.float_] = np.concatenate(
    #         (d, angle_diff, velocity_diff), axis=1
    #     )

    #     reward[np.where(differences[:, 0] < 1)] = 1
    #     reward[
    #         np.where(
    #             (differences[:, 1] < np.deg2rad(10))
    #             & (differences[:, 1] > np.deg2rad(-10))
    #             & (differences[:, 0] < 1)
    #         )
    #     ] = 2
    #     reward[
    #         np.where(
    #             (differences[:, 2] < 1)
    #             & (differences[:, 1] < np.deg2rad(10))
    #             & (differences[:, 1] > np.deg2rad(-10))
    #             & (differences[:, 0] < 1)
    #         )
    #     ] = 3

    #     return np.float_(reward) if len(reward) == 1 else reward

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )

        # if crashed:
        #     print("Crashed")
        # elif success:
        #     print("Success")
        # else:
        #     pass

        return bool(crashed or success)

    # def _is_success(
    #     self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    # ) -> np.bool_:
    #     return self.compute_reward(achieved_goal, desired_goal, {}) >= 3

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


class QuasiContinuousAction(ActionType):
    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 4, np.pi / 4)
    """Steering angle range: [-x, x], in rad."""

    def __init__(
        self,
        env: "AbstractEnv",
        acceleration_range: Optional[tuple[float, float]] = None,
        steering_range: Optional[tuple[float, float]] = None,
        speed_range: Optional[tuple[float, float]] = None,
        longitudinal: bool = True,
        lateral: bool = True,
        dynamical: bool = False,
        clip: bool = True,
        **kwargs,
    ) -> None:
        """
        Create a continuous action space.

        Parameters
        ----------

        env: the environment
        acceleration_range: the range of acceleration values [m/s²]
        steering_range: the range of steering values [rad]
        speed_range: the range of reachable speeds [m/s]
        longitudinal: enable throttle control
        lateral: enable steering control
        dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = (
            acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        )
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError(
                "Either longitudinal and/or lateral control must be enabled"
            )
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    def act(self, action: np.ndarray) -> None:
        if self.clip:
            action = np.clip(action, -1, 1)
        if self.speed_range:
            self.controlled_vehicle.MIN_SPEED, self.controlled_vehicle.MAX_SPEED = (
                self.speed_range
            )
        if self.longitudinal and self.lateral:
            self.controlled_vehicle.act(
                {
                    "acceleration": utils.lmap(
                        action[0], [-1, 1], self.acceleration_range
                    ),
                    "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
                }
            )
        elif self.longitudinal:
            self.controlled_vehicle.act(
                {
                    "acceleration": utils.lmap(
                        action[0], [-1, 1], self.acceleration_range
                    ),
                    "steering": 0,
                }
            )
        elif self.lateral:
            self.controlled_vehicle.act(
                {
                    "acceleration": 0,
                    "steering": utils.lmap(action[0], [-1, 1], self.steering_range),
                }
            )
        self.last_action = action


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

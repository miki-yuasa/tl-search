from dataclasses import dataclass
from typing import Literal, Any, TypedDict, NotRequired

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import matplotlib.pyplot as plt


class World(TypedDict):
    agent_color: str
    adversary_color: str
    goal_color: str
    landmark_color: str


default_world: World = {
    "agent_color": "blue",
    "adversary_color": "red",
    "goal_color": "green",
    "landmark_color": "yellow",
}


class Object(TypedDict):
    size: float
    color: str
    pos: NDArray[np.float64]
    vel_polar: NotRequired[NDArray[np.float64]]
    vel_cartesian: NotRequired[NDArray[np.float64]]


class Info(TypedDict):
    agent: Object
    adversaries: Object
    goals: Object
    landmarks: Object


class Field:
    def __init__(
        self,
        world: World,
        agent_size: float,
        adversary_size: float,
        goal_size: float,
        landmark_size: float,
        num_adversaries: int,
        num_goals: int,
        num_landmarks: int,
    ) -> None:
        self.num_adversaries: int = num_adversaries
        self.num_goals: int = num_goals
        self.num_landmarks: int = num_landmarks
        self.agent_size: float = agent_size
        self.adversary_size: float = adversary_size
        self.goal_size: float = goal_size
        self.landmark_size: float = landmark_size

        self.info: Info = {
            "agent": {
                "color": world["agent_color"],
                "pos": np.array([0, 0]),
                "vel_polar": np.array([0, 0]),
                "vel_cartesian": np.array([0, 0]),
                "size": agent_size,
            },
            "adversaries": {
                "color": world["adversary_color"],
                "pos": np.array([[0, 0] for _ in range(num_adversaries)]),
                "vel_polar": np.array([[0, 0] for _ in range(num_adversaries)]),
                "vel_cartesian": np.array([[0, 0] for _ in range(num_adversaries)]),
                "size": adversary_size,
            },
            "goals": {
                "color": world["goal_color"],
                "pos": np.array([[0, 0] for _ in range(num_goals)]),
                "size": goal_size,
            },
            "landmarks": {
                "color": world["landmark_color"],
                "pos": np.array([[0, 0] for _ in range(num_landmarks)]),
                "size": landmark_size,
            },
        }

    @staticmethod
    def compute_new_pos(
        current_pos: NDArray[np.float64], vel_polar: NDArray[np.float64], bound: float
    ) -> NDArray[np.float64]:
        vel_cartesian = np.array(
            [vel_polar[0] * np.cos(vel_polar[1]), vel_polar[0] * np.sin(vel_polar[1])]
        )

        new_pos: NDArray[np.float64] = current_pos + vel_cartesian
        new_pos_clipped: NDArray[np.float64] = np.clip(new_pos, -bound, bound)

        return new_pos_clipped

    @staticmethod
    def cartesian2poler(vel_cartesian: NDArray[np.float64]) -> NDArray[np.float64]:
        vel_polar = np.array(
            [
                np.linalg.norm(vel_cartesian),
                np.arctan2(vel_cartesian[1], vel_cartesian[0]),
            ]
        )
        return vel_polar

    @staticmethod
    def polar2cartesian(vel_polar: NDArray[np.float64]) -> NDArray[np.float64]:
        vel_cartesian = np.array(
            [vel_polar[0] * np.cos(vel_polar[1]), vel_polar[0] * np.sin(vel_polar[1])]
        )
        return vel_cartesian

    @staticmethod
    def compute_cartesian_vel(
        new_pos: NDArray[np.float64], current_pos: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        vel_cartesian = new_pos - current_pos
        return vel_cartesian

    @staticmethod
    def compute_polar_vel(
        new_pos: NDArray[np.float64], current_pos: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        vel_cartesian = new_pos - current_pos
        vel_polar = Field.cartesian2poler(vel_cartesian)
        return vel_polar

    def move_agent(self, action: NDArray[np.float64], bound: float) -> None:
        self.info["agent"]["pos"] = Field.compute_new_pos(
            self.info["agent"]["pos"], action, bound
        )
        self.info["agent"]["vel_polar"] = action
        self.info["agent"]["vel_cartesian"] = Field.polar2cartesian(action)

    def move_adversaries(self, bound: float) -> None:
        assert "vel_polar" in self.info["adversaries"]
        assert "vel_cartesian" in self.info["adversaries"]

        for i in range(self.num_adversaries):
            self.info["adversaries"]["pos"][i] = Field.compute_new_pos(
                self.info["adversaries"]["pos"][i],
                self.info["adversaries"]["vel_polar"][i],
                bound,
            )
            self.info["adversaries"]["vel_cartesian"][i] = Field.polar2cartesian(
                self.info["adversaries"]["vel_polar"][i]
            )


class SimpleTagEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        world: World = default_world,
        max_steps: int = 50,
        agent_size: float = 1.0,
        adversary_size: float = 1.0,
        goal_size: float = 3.0,
        landmark_size: float = 3.0,
        bound: float = 20.0,
        num_adversaries: int = 3,
        num_goals: int = 1,
        num_landmarks: int = 2,
        agent_max_speed: float = 1.0,
        adversary_max_speed: float = 0.5,
        agent_clearance: float = 1.0,
        goal_reward: float = 10.0,
        obstacle_reward: float = -10.0,
        adversary_reward: float = 10.0,
        step_reward: float = -0.01,
    ):
        self.world: World = world
        self.max_steps: int = max_steps
        self.num_adversaries: int = num_adversaries
        self.agent_size: float = agent_size
        self.adversary_size: float = adversary_size
        self.agent_max_speed: float = agent_max_speed
        self.adversary_max_speed: float = adversary_max_speed
        self.num_goals: int = num_goals
        self.goal_size: float = goal_size
        self.num_landmarks: int = num_landmarks
        self.landmark_size: float = landmark_size
        self.bound: float = bound
        self.agent_clearance: float = agent_clearance
        self.goal_reward: float = goal_reward
        self.obstacle_reward: float = obstacle_reward
        self.adversary_reward: float = adversary_reward
        self.step_reward: float = step_reward

        self.field: Field = Field(
            world,
            agent_size,
            adversary_size,
            goal_size,
            landmark_size,
            num_adversaries,
            num_goals,
            num_landmarks,
        )

        self.action_space = Box(
            low=np.array([0, 0]),
            high=np.array([agent_max_speed, np.pi * 2]),
            shape=(2,),
        )
        self.observation_space = Dict(
            {
                "self_vel": Box(
                    low=np.array([0, 0]),
                    high=np.array([agent_max_speed, np.pi * 2]),
                    shape=(2,),
                ),
                "self_pos": Box(
                    low=np.array([0, 0]),
                    high=np.array([bound, bound]),
                    shape=(2,),
                ),
                "adversary_vel": Box(
                    low=np.array([[0, 0] for _ in range(num_adversaries)]),
                    high=np.array(
                        [
                            [adversary_max_speed, np.pi * 2]
                            for _ in range(num_adversaries)
                        ]
                    ),
                    shape=(num_adversaries, 2),
                ),
                "adversary_rel_pos": Box(
                    low=np.array([[-bound, -bound] for _ in range(num_adversaries)]),
                    high=np.array([[bound, bound] for _ in range(num_adversaries)]),
                    shape=(num_adversaries, 2),
                ),
                "goal_rel_pos": Box(
                    low=np.array([[-bound, -bound] for _ in range(num_goals)]),
                    high=np.array([[bound, bound] for _ in range(num_goals)]),
                    shape=(num_goals, 2),
                ),
                "landmark_rel_pos": Box(
                    low=np.array([[-bound, -bound] for _ in range(num_landmarks)]),
                    high=np.array([[bound, bound] for _ in range(num_landmarks)]),
                    shape=(num_landmarks, 2),
                ),
            }
        )

    def _get_obs(self) -> dict[str, Any]:
        assert "vel_polar" in self.field.info["agent"]
        self_vel = self.field.info["agent"]["vel_polar"]
        self_pos = self.field.info["agent"]["pos"]
        assert "vel_polar" in self.field.info["adversaries"]
        adversary_vel = self.field.info["adversaries"]["vel_polar"]
        adversary_pos = self.field.info["adversaries"]["pos"]
        goal_pos = self.field.info["goals"]["pos"]
        landmark_pos = self.field.info["landmarks"]["pos"]

        adversary_rel_pos = adversary_pos - self_pos
        goal_rel_pos = goal_pos - self_pos
        landmark_rel_pos = landmark_pos - self_pos

        return {
            "self_vel": self_vel,
            "self_pos": self_pos,
            "adversary_vel": adversary_vel,
            "adversary_rel_pos": adversary_rel_pos,
            "goal_rel_pos": goal_rel_pos,
            "landmark_rel_pos": landmark_rel_pos,
        }

    def step(self, action):
        pass

    def reset(self):
        self.steps: int = 0

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self):
        if self.viewer is None:
            self.viewer = plt.figure()
        return self.viewer

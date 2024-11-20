import numpy as np
from numpy.typing import NDArray

import gymnasium
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder, RenderConf
from safety_gymnasium.utils.task_utils import get_task_class_name
from safety_gymnasium import tasks
from safety_gymnasium.utils.common_utils import quat2zalign

from tl_search.envs import tl_safety_tasks


class CustomBuilder(Builder):
    def __init__(
        self,
        task_id: str,
        max_episode_steps: int = 500,
        config: dict | None = None,
        render_mode: str | None = None,
        width: int = 256,
        height: int = 256,
        camera_id: int | None = None,
        camera_name: str | None = None,
        ignore_cost: bool = False,
    ) -> None:
        gymnasium.utils.EzPickle.__init__(
            self,
            config=config,
            task_id=task_id,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            camera_name=camera_name,
            ignore_cost=ignore_cost,
        )

        self.task_id: str = task_id
        self.config: dict = config
        self._seed: int = None
        self._setup_simulation()

        self.first_reset: bool = None
        self.steps: int = None
        self.cost: float = None
        self.terminated: bool = True
        self.truncated: bool = False

        self.render_parameters = RenderConf(
            render_mode, width, height, camera_id, camera_name
        )

        self.max_episode_steps: int = max_episode_steps
        self.ignore_cost: bool = ignore_cost

    def _get_task(self) -> BaseTask:
        class_name = get_task_class_name(self.task_id)
        if hasattr(tasks, class_name):
            task_class = getattr(tasks, class_name)
        elif hasattr(tl_safety_tasks, class_name):
            task_class = getattr(tl_safety_tasks, class_name)
        else:
            raise NotImplementedError(f"Task={class_name} not implemented.")

        task = task_class(config=self.config)

        task.build_observation_space()
        return task

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict]:
        """Take a step and return observation, reward, cost, terminated, truncated, info."""
        assert not self.done, "Environment must be reset before stepping."
        action = np.array(action, copy=False)  # cast to ndarray
        if action.shape != self.action_space.shape:  # check action dimension
            raise ValueError("Action dimension mismatch")

        info = {}

        exception = self.task.simulation_forward(action)
        if exception:
            self.truncated = True

            reward = self.task.reward_conf.reward_exception
            info["cost_exception"] = 1.0
        else:
            # Reward processing
            reward = self._reward()

            # Constraint violations
            info.update(self._cost())

            cost = info["cost_sum"]

            self.task.specific_step()

            # Goal processing
            if self.task.goal_achieved:
                info["goal_met"] = True
                self.terminated = True

            if info["cost_hazards"] > 0:
                self.terminated = True

        # termination of death processing
        if not self.task.agent.is_alive():
            self.terminated = True

        # Timeout
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            self.truncated = True  # Maximum number of steps in an episode reached

        if self.render_parameters.mode == "human":
            self.render()

        if not self.ignore_cost:
            reward -= cost
        else:
            pass

        info.update(
            {
                "is_success": self._is_success(),
            }
        )

        return self.task.obs(), reward, self.terminated, self.truncated, info

    def _is_success(self) -> bool:
        return self.task.dist_goal() <= self.task.goal.size

    def _reward(self) -> float:
        """Calculate the current rewards.

        Call exactly once per step.
        """
        reward = self.task.calculate_reward()

        # Intrinsic reward for uprightness
        if self.task.reward_conf.reward_orientation:
            zalign = quat2zalign(
                self.task.data.get_body_xquat(
                    self.task.reward_conf.reward_orientation_body
                ),
            )
            reward += self.task.reward_conf.reward_orientation_scale * zalign

        # Clip reward
        reward_clip = self.task.reward_conf.reward_clip
        if reward_clip:
            in_range = -reward_clip < reward < reward_clip
            if not in_range:
                reward = np.clip(reward, -reward_clip, reward_clip)

        return reward

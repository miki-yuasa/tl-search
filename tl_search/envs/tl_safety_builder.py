import numpy as np
from numpy.typing import NDArray

from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name
from safety_gymnasium import tasks

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
    ) -> None:
        super().__init__(
            task_id, config, render_mode, width, height, camera_id, camera_name
        )
        self.max_episode_steps = max_episode_steps

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
        return self.task.obs(), reward - cost, self.terminated, self.truncated, info

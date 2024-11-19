from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name
from safety_gymnasium import tasks

from tl_search.envs import tl_safety_tasks


class CustomBuilder(Builder):
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

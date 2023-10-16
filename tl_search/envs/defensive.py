import gym
from gym import spaces
import numpy as np
from numpy.typing import NDArray

from tl_search.common.typing import MapObjectIDs


class DefensiveEnv(gym.Env):
    """
    Defensive policy environment that follwos gym interface
    """

    def __init__(
        self, n_in: int, init_map: NDArray, map_object_ids: MapObjectIDs
    ) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=n_in, dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

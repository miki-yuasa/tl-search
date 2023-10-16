import random
from typing import cast
import gym
from gym import spaces
import numpy as np
from numpy.typing import NDArray

from tl_search.map.utils import parse_map
from policies.known_policy import AStarPolicy

from tl_search.common.typing import FixedMapLocations, Location, MapObjectIDs


class BaseEnv(gym.Env):
    """
    Defensive policy environment that follwos gym interface
    """

    def __init__(
        self, n_in: int, init_map: NDArray, map_object_ids: MapObjectIDs
    ) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(n_in,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.map_object_ids: MapObjectIDs = map_object_ids

        self.fixed_map_locs: FixedMapLocations = cast(
            FixedMapLocations, parse_map(init_map, self.map_object_ids)
        )

        (
            self.blue_flag,
            self.red_flag,
            self.blue_territory,
            self.red_territory,
            self.obstacles,
            self.walls,
        ) = self.fixed_map_locs

        self.blue_agent: Location = random.choice(self.blue_territory)
        self.red_agent: Location = random.choice(self.red_territory)

        red_policy = AStarPolicy(
            self.red_agent,
            self.blue_flag,
            init_map,
        )
        self.red_path = red_policy.path
        self.red_path_counter = 0

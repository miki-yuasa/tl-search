import random, sys
from heapq import heapify, heappush, heappop
from typing import Union, cast

import numpy as np
from numpy.typing import NDArray

from tl_search.map.utils import (
    manhattan_distance,
    read_map,
    tuples2locs,
    parse_map,
    sample_agent_locs,
)
from tl_search.common.typing import (
    Action,
    ActionProb,
    MapObjectIDs,
    Location,
    AStarNode,
    FixedMapLocations,
    MapLocations,
    PolicyMode,
)
from tl_search.policies.heuristic import a_star


class DefensivePolicy:
    def __init__(self, map_locs: MapLocations, map: NDArray):
        goals: list[Location]
        red_loc = map_locs.red_agent
        blue_loc = map_locs.blue_agent
        obstacles = map_locs.obstacles
        walls = map_locs.walls
        if red_loc.x >= 5:
            goals = [red_loc]
        elif red_loc.y <= 4:
            goals = tuples2locs(
                [
                    (0, 5),
                    (0, 6),
                    (1, 5),
                    (1, 6),
                ]
            )
        else:
            goals = tuples2locs(
                [
                    (9, 5),
                    (9, 6),
                    (8, 5),
                    (8, 6),
                ]
            )

        action: Action

        if blue_loc in goals:
            action = 0  # stay
        elif blue_loc in obstacles + walls or red_loc in [map_locs.blue_flag]:
            action = -1
        else:
            manhattan_distances: list[int] = [
                manhattan_distance(blue_loc, goal) for goal in goals
            ]
            min_dist_inds: list[int] = [
                i
                for i, dist in enumerate(manhattan_distances)
                if dist == min(manhattan_distances)
            ]
            goal: Location = goals[random.choice(min_dist_inds)]
            path = a_star(blue_loc, goal, map)

            next_loc = Location(*path[1])
            dx: int = next_loc.x - blue_loc.x
            dy: int = next_loc.y - blue_loc.y

            if dx == -1 and dy == 0:
                action = 1  # left
            elif dx == 0 and dy == 1:
                action = 2  # down
            elif dx == 1 and dy == 0:
                action = 3  # right
            elif dx == 0 and dy == -1:
                action = 4  # up
            else:
                print(
                    "Error: invalid blue agent movement.",
                    file=sys.stderr,
                )
                sys.exit(1)

        action_prob: ActionProb = cast(
            ActionProb,
            (0.0, 0.0, 0.0, 0.0, 0.0)
            if action == -1
            else tuple([1.0 if i == action else 0.0 for i in range(5)]),
        )

        self.action_prob: NDArray = np.array(action_prob)
        self.path: list[Location] = path


class KnownPolicy:
    def __init__(self, policy_filename: str, map_object_ids: MapObjectIDs) -> None:
        self.map: NDArray = read_map(policy_filename)
        self.fixed_map_locs: FixedMapLocations = cast(
            FixedMapLocations, parse_map(self.map, map_object_ids)
        )
        self.map_object_ids = map_object_ids
        self.policy_mode: PolicyMode = "known"

    def sample(
        self, num_sampling: int, is_from_all_territories: bool = True
    ) -> tuple[list[MapLocations], list[NDArray]]:
        map_locs_list = [
            sample_agent_locs(self.fixed_map_locs, is_from_all_territories)
            for _ in range(num_sampling)
        ]
        ori_action_probs_np = [
            DefensivePolicy(
                map_locs,
                self.map,
            ).action_prob
            for map_locs in map_locs_list
        ]

        return (map_locs_list, ori_action_probs_np)

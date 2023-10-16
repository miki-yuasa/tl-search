import random
from typing import cast, Union

import numpy as np
from numpy.typing import NDArray

from numpy.lib.npyio import NpzFile

from tl_search.map.utils import parse_map
from tl_search.common.typing import MapObjectIDs, MapLocations, PolicyMode


class UnknownPolicy:
    def __init__(self, policy_filename: str, map_object_ids: MapObjectIDs) -> None:
        data: NpzFile = np.load(policy_filename)
        states: NDArray = data["s0"]
        actions: NDArray = data["a"]
        # locations: NDArray = data["loc"]
        # enemy_location: NDArray = data["loc_e"]
        # subpolicy: NDArray = data["sp"]

        map_locs_list: list[Union[MapLocations, None]] = [
            cast(Union[MapLocations, None], parse_map(state, map_object_ids, True))
            for state in states
        ]
        self.map_locs_list: list[MapLocations] = [
            cast(MapLocations, map_locs)
            for map_locs in map_locs_list
            if map_locs != None
        ]
        self.action_probs: list[NDArray] = [
            np.array(tuple([1.0 if i == action else 0.0 for i in range(5)]))
            for map_locs, action in zip(map_locs_list, list(actions[:, 0]))
            if map_locs != None
        ]

        self.map: NDArray = states[0]
        self.map_object_ids: MapObjectIDs = map_object_ids

        self.policy_mode: PolicyMode = "unknown"

    def sample(self, num_sampling: int) -> tuple[list[MapLocations], list[NDArray]]:
        map_locs_list: list[MapLocations]
        ori_action_probs_np: list[NDArray]

        if num_sampling < len(self.map_locs_list):
            map_locs_list, ori_action_probs_np = zip(
                *random.sample(
                    list(zip(self.map_locs_list, self.action_probs)),
                    num_sampling,
                )
            )

        else:
            map_locs_list = self.map_locs_list
            ori_action_probs_np = self.action_probs

        return (map_locs_list, ori_action_probs_np)


if __name__ == "__main__":
    policy_filename: str = "./policies/data/CTF10_1v1_SubPolicy17.npz"
    policy = UnknownPolicy(policy_filename, MapObjectIDs(5, -5, 2, -2, None, None, 8))

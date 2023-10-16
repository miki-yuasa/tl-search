from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from tl_search.tl.constants import (
    known_policy_map_object_ids,
    unknown_policy_map_object_ids,
)
from tl_search.tl.environment import Environment
from tl_search.evaluation.extractor import get_action_distribution
from tl_search.map.utils import read_map, sample_agent_locs
from policies.known_policy import KnownPolicy
from policies.unknown_policy import UnknownPolicy
from tl_search.common.typing import (
    FixedMapLocations,
    MapLocations,
    PolicyMode,
    MapObjectIDs,
)


def get_map_object_ids(policy_mode: PolicyMode) -> MapObjectIDs:
    map_object_ids: MapObjectIDs = (
        known_policy_map_object_ids
        if policy_mode == "known"
        else unknown_policy_map_object_ids
    )

    return map_object_ids


def get_init_map(
    policy_mode: PolicyMode, knwon_map_file: str, unknown_policy: UnknownPolicy
) -> NDArray:
    init_map: NDArray = (
        read_map(knwon_map_file) if policy_mode == "known" else unknown_policy.init_map
    )
    return init_map


def sample_learned_policy(
    fixed_map_locs: FixedMapLocations,
    target: PPO,
    num_sampling: int,
    is_from_all_territories: bool,
):
    map_locs_list: list[MapLocations] = [
        sample_agent_locs(fixed_map_locs, is_from_all_territories)
        for _ in range(num_sampling)
    ]

    env = cast(Environment, target.env.envs[0].env)

    action_probs: list[NDArray] = [
        get_action_distribution(
            cast(ActorCriticPolicy, target.policy),
            torch.tensor(
                np.array(
                    [env.reset_with_agent_locs(map_locs.blue_agent, map_locs.red_agent)]
                )
            ).to(target.device),
        )
        for map_locs in map_locs_list
    ]

    return map_locs_list, action_probs

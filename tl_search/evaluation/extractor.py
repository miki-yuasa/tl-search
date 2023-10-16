import json, os
from typing import Callable, cast

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

import numpy as np
from numpy.typing import NDArray

from torch import Tensor
import torch

from tl_search.tl.environment import Environment, create_env
from tl_search.common.typing import (
    KLDivLogger,
    MapLocations,
    MapObjectIDs,
    ObsProp,
    PolicyMode,
)
from tl_search.tl.tl_parser import atom_tl_ob2rob
from tl_search.tl.transition import find_trap_transitions


def retrieve_rewards(
    data: KLDivLogger,
    policy_mode: PolicyMode,
    experiment: int,
    model_filenames: list[str],
    init_map: NDArray,
    policy_map_object_ids: MapObjectIDs,
    obs_info: list[tuple[str, list[str], Callable]],
    atom_prop_dict_all: dict[str, str],
    save_file: bool = False,
) -> tuple[list[float], list[float]]:
    obs_props_all: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]

    mean_rewards: list[float] = []
    std_rewards: list[float] = []
    for tl_spec, filename in zip(data.tl_specs, model_filenames):
        print("Evaluating: {}".format(tl_spec))
        env = create_env(
            tl_spec, init_map, policy_map_object_ids, obs_props_all, atom_prop_dict_all
        )
        env_monitored = Monitor(env)
        model = PPO.load(filename, env_monitored)
        mean_reward, std_reward = evaluate_policy(
            model, model.get_env(), n_eval_episodes=10
        )
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)

    d = {"mean_rewards": mean_rewards, "std_rewards": std_rewards}

    if save_file:
        savename: str = "rewards_{}_{}.json".format(policy_mode, experiment)
        filepath: str = os.path.join(os.path.dirname(__file__), "..", "out", savename)
        with open(filepath, "w") as f:
            json.dump(d, f, indent=4)
    else:
        pass

    return (mean_rewards, std_rewards)


def get_action_distribution(policy: ActorCriticPolicy, obs: Tensor) -> NDArray:
    features = policy.extract_features(obs)
    latent_pi, _ = policy.mlp_extractor(features)
    distribution: CategoricalDistribution = cast(
        CategoricalDistribution,
        policy._get_action_dist_from_latent(latent_pi),
    )
    tl_action_prob: NDArray = distribution.distribution.probs[0].cpu().detach().numpy()

    return tl_action_prob


def get_agent_action_distributions(
    model: PPO, map_locs_list: list[MapLocations]
) -> tuple[NDArray, NDArray]:
    env: Environment = model.env.envs[0].env
    aut = env.aut

    action_probs: list[NDArray] = []
    trap_mask: list[int] = []
    print("Getting action distributions...")
    for i, map_locs in enumerate(map_locs_list):
        # print(f"Map {i+1}/{len(map_locs_list)}")
        action_prob: NDArray = get_action_distribution(
            model.policy,
            torch.tensor(
                np.array(
                    [env.reset_with_agent_locs(map_locs.blue_agent, map_locs.red_agent)]
                )
            ).to(model.device),
        )

        action_probs.append(action_prob)

        atom_rob_dict, _ = atom_tl_ob2rob(aut, map_locs)
        trap_mask.append(0 if find_trap_transitions(atom_rob_dict, aut) else 1)

    return np.array(action_probs), np.array(trap_mask)


def get_agent_actions(
    model: DQN, map_locs_list: list[MapLocations]
) -> tuple[NDArray, NDArray]:
    env: Environment = model.env.envs[0].env
    aut = env.aut

    actions: list[int] = []
    trap_mask: list[int] = []
    for map_locs in map_locs_list:
        action: int = model.predict(
            np.array(
                [env.reset_with_agent_locs(map_locs.blue_agent, map_locs.red_agent)]
            ),
            deterministic=True,
        )[0][0]

        actions.append(action)

        atom_rob_dict, _ = atom_tl_ob2rob(aut, map_locs)
        trap_mask.append(0 if find_trap_transitions(atom_rob_dict, aut) else 1)

    return np.array(actions), np.array(trap_mask)

from math import floor
from typing import cast

import numpy as np
from numpy.typing import NDArray

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from tl_search.map.utils import manhattan_distance

from tl_search.tl.constants import fight_range
from tl_search.tl.environment import act, Environment
from tl_search.evaluation.extractor import get_action_distribution
from policies.known_policy import (
    DefensivePolicy,
    KnownPolicy,
)
from tl_search.policies.heuristic import a_star
from tl_search.common.typing import ActionProb, Location, Action


def simulate_known_policy(
    blue_agent: Location, red_agent: Location, known_policy: KnownPolicy, max_iter: int
) -> tuple[list[Action], list[NDArray]]:
    red_path: list[Location] = a_star(
        red_agent, known_policy.fixed_map_locs.blue_flag, known_policy.map
    )
    blue_path: list[Location] = [blue_agent]

    action: Action
    actions: list[Action] = []
    action_probs: list[NDArray] = []

    for cnt in range(max_iter):
        if cnt == 0:
            pass
        else:
            blue_agent = act(blue_agent, action)
            blue_path.append(blue_agent)
            if manhattan_distance(red_path[cnt], blue_agent) <= fight_range:
                break
            else:
                pass

        action_prob = DefensivePolicy(
            blue_agent, red_path[cnt], known_policy.map
        ).action_prob
        action_probs.append(action_prob)

        action = np.argmax(action_prob)
        actions.append(action)

    return actions, action_probs


def simulate_learned_policy(
    model: PPO,
    env: Environment,
    blue_agent: Location,
    red_agent: Location,
    max_iter: int,
) -> tuple[list[Action], list[NDArray]]:
    actions: list[Action] = []
    action_probs: list[NDArray] = []

    obs: NDArray = env.reset_with_agent_locs(blue_agent, red_agent)

    for i in range(max_iter):
        policy: ActorCriticPolicy = cast(ActorCriticPolicy, model.policy)
        action_prob: NDArray = get_action_distribution(
            policy, torch.tensor([obs]).to(model.device)
        )
        action_probs.append(action_prob)

        action = cast(Action, np.argmax(action_prob))
        actions.append(action)
        obs, _, done, _ = env.step(action)

        if done:
            break
        else:
            pass

    return actions, action_probs


def create_comparison_domain(
    actions: list[Action], action_probs: list[NDArray], steps: int
) -> NDArray:
    grids: int = int((2 * steps + 1) + 2 * (np.sum(np.array(range(1, steps * 2, 2)))))
    domain: list[float] = [0 for i in range(grids)]

    row_diff_max: int = 2 * steps
    center_ind: int = floor(grids / 2)
    from_center: int = 0

    for action, action_prob in zip(actions, action_probs):
        row_diff: int = row_diff_max - 2 * abs(from_center)

        domain[center_ind] += action_prob[0]  # stay
        domain[center_ind - 1] += action_prob[1]  # left
        domain[center_ind + row_diff] += action_prob[2]  # down
        domain[center_ind + 1] += action_prob[3]  # right
        domain[center_ind - row_diff] += action_prob[4]  # up

        if action == 2:
            from_center -= 1
            center_ind += row_diff
        elif action == 4:
            from_center += 1
            center_ind -= row_diff
        elif action == 0:
            pass
        elif action == 1:
            center_ind -= 1
        elif action == 3:
            center_ind += 1
        else:
            raise Exception

    return np.array([domain])

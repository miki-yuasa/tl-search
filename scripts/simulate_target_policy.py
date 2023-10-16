import random

import numpy as np

from tl_search.map.utils import manhattan_distance
from tl_search.policies.heuristic import a_star
from tl_search.tl.constants import known_policy, fight_range
from tl_search.tl.environment import act
from tl_search.policies.known_policy import DefensivePolicy
from tl_search.evaluation.visualize import create_animation
from tl_search.common.typing import Action, Location

savename: str = "known_target.gif"

blue_agent: Location = random.choice(known_policy.fixed_map_locs.blue_territory)
red_agent: Location = random.choice(known_policy.fixed_map_locs.red_territory)

red_path: list[Location] = a_star(
    red_agent, known_policy.fixed_map_locs.blue_flag, known_policy.map
).path
blue_path: list[Location] = [blue_agent]

done: bool = False
cnt: int = 0
action: Action

while not done:
    if cnt == 0:
        pass
    else:
        blue_agent = act(blue_agent, action)
        blue_path.append(blue_agent)
        if manhattan_distance(red_path[cnt], blue_agent) <= fight_range:
            done = True
            break
        else:
            pass

    action_prob = DefensivePolicy(
        blue_agent, red_path[cnt], known_policy.map
    ).action_prob
    action = np.argmax(action_prob)
    cnt += 1

create_animation(
    blue_path, red_path, known_policy.map, known_policy.map_object_ids, savename
)

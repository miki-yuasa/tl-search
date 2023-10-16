import itertools
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.distributions import Distribution
from tl_search.common.typing import Location

from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.envs.typing import FieldObj
from tl_search.tl.tl_parser import atom_tl_ob2rob
from tl_search.tl.transition import find_trap_transitions


def generate_possible_states(sample_env: TLMultigrid) -> list[FieldObj]:
    possible_locs: list[Location] = list(
        set(
            [
                Location(*loc)
                for loc in itertools.product(
                    range(sample_env._field_map.shape[0]),
                    range(sample_env._field_map.shape[1]),
                )
            ]
        )
        - set(sample_env._fixed_obj.obstacle)
    )

    red_agent_status: tuple[bool, bool] = (False, True)

    all_states = itertools.product(possible_locs, possible_locs, red_agent_status)
    possible_states: list[FieldObj] = []

    for blue_agent, red_agent, is_red_agent_defeated in all_states:
        if blue_agent == red_agent:
            if not is_red_agent_defeated:
                continue
            else:
                pass
        else:
            pass

        sample_env.reset(blue_agent, red_agent, is_red_agent_defeated)
        possible_states.append(sample_env._field)

    return possible_states


def get_action_distributions(
    model: PPO, env: TLMultigrid, field_obj_list: list[FieldObj]
) -> tuple[NDArray, NDArray]:
    aut = env.aut

    action_probs: list[NDArray] = []
    trap_mask: list[int] = []
    for field_obj in field_obj_list:
        obs, _ = env.reset(
            field_obj.blue_agent, field_obj.red_agent, field_obj.is_red_agent_defeated
        )
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dis: Distribution = model.policy.get_distribution(obs_tensor)
        probs_np: NDArray = dis.distribution.probs[0].cpu().detach().numpy()

        action_probs.append(probs_np)

        atom_rob_dict, _ = atom_tl_ob2rob(aut, field_obj)
        trap_mask.append(0 if find_trap_transitions(atom_rob_dict, aut) else 1)

    env.close()

    return np.array(action_probs), np.array(trap_mask)

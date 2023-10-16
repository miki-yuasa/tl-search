import random

import torch

from tl_search.common.typing import Location


def elementwise_cbf(
    actions: torch.Tensor, blue_locitions: list[tuple[int, int]], aut_states, map
):
    h, w = map.shape

    for index, (action, blue_loc, aut_state) in enumerate(
        zip(actions, blue_locitions, aut_states)
    ):
        blue_x, blue_y = blue_loc

        permitted_actions = [0]

        # left
        if blue_x - 1 < 0:
            pass
        elif map[blue_x - 1, blue_y] == 8:
            pass
        elif aut_state == 0:
            if map[blue_x - 1, blue_y] == 1:
                pass
            else:
                permitted_actions.append(1)
        else:
            permitted_actions.append(1)

        # down
        if blue_y + 1 >= h:
            pass
        elif map[blue_x, blue_y + 1] == 8:
            pass
        elif aut_state == 0:
            if map[blue_x, blue_y + 1] == 1:
                pass
            else:
                permitted_actions.append(2)
        else:
            permitted_actions.append(2)

        # right
        if blue_x + 1 >= w:
            pass
        elif map[blue_x + 1, blue_y] == 8:
            pass
        elif aut_state == 0:
            if map[blue_x + 1, blue_y] == 1:
                pass
            else:
                permitted_actions.append(3)
        else:
            permitted_actions.append(3)

        # up
        if blue_y - 1 < 0:
            pass
        elif map[blue_x, blue_y - 1] == 8:
            pass
        elif aut_state == 0:
            if map[blue_x, blue_y - 1] == 1:
                pass
            else:
                permitted_actions.append(4)
        else:
            permitted_actions.append(4)

        if action in permitted_actions:
            pass
        else:
            actions[index] = random.choice(permitted_actions)

    return actions


def control_barrier_function(
    action_probs: torch.Tensor, blue_locitions: list[Location], map
):
    h, w = map.shape

    action_probs_copy = action_probs.detach().clone()

    if True in torch.isnan(action_probs):
        print("NaN in action probs")

    for index, blue_loc in enumerate(blue_locitions):
        blue_x, blue_y = blue_loc
        action_prob = action_probs[index].detach().clone()

        if blue_x - 1 < 0:
            action_prob[1] = 0
        elif map[blue_x - 1, blue_y] == 8:
            action_prob[1] = 0
        else:
            pass

        if blue_y + 1 >= h:
            action_prob[2] = 0
        elif map[blue_x, blue_y + 1] == 8:
            action_prob[2] = 0
        else:
            pass

        if blue_x + 1 >= w:
            action_prob[3] = 0
        elif map[blue_x + 1, blue_y] == 8:
            action_prob[3] = 0
        else:
            pass

        if blue_y - 1 < 0:
            action_prob[4] = 0
        elif map[blue_x, blue_y - 1] == 8:
            action_prob[4] = 0
        else:
            pass

        action_prob_tmp = action_prob / action_prob.sum(0)
        if True in torch.isnan(action_prob_tmp):
            # print("NaN in action probs")
            pass
        else:
            action_probs[index] = action_prob_tmp

    return action_probs

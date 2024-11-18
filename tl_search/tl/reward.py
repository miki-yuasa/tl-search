import sys, math, random
from collections import deque

import numpy as np

from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import tl2rob
from tl_search.common.typing import RobustnessCounter, AutomatonStateCounter, Transition


def transition_robustness(
    transitions: list[Transition],
    atom_rob_dict: dict[str, float],
) -> tuple[list[float], list[float], list[float]]:
    robs: list[float] = []
    non_trap_robs: list[float] = []
    trap_robs: list[float] = []
    for trans in transitions:
        rob: float = tl2rob(trans.condition, atom_rob_dict)
        robs.append(rob)
        if trans.is_trapped_next:
            trap_robs.append(rob)
        else:
            non_trap_robs.append(rob)

    return robs, non_trap_robs, trap_robs


def tl_reward(
    atom_rob_dict: dict[str, float],
    aut: TLAutomaton,
    curr_aut_state: int,
    dense_reward: bool = False,
    terminal_state_reward: float = 5,
) -> tuple[float, int]:
    """
    Calculate the reward of the step from a given automaton.
    Parameters:

    atom_robs: robustnesses from atomic predicates
    aut: automaton from a TL spec.
    curr_aut_state: current automaton state
    Returns:

    reward (float): reward of the step based on the MDP and automaton states.
    next_aut_state (int): the resultant automaton state
    """

    if curr_aut_state in aut.goal_states:
        return (terminal_state_reward, curr_aut_state)
    elif curr_aut_state in aut.trap_states:
        return (-terminal_state_reward, curr_aut_state)
    else:

        curr_edge = aut.edges[curr_aut_state]
        transitions = curr_edge.transitions

        # Calculate robustnesses of the transitions
        robs, non_trap_robs, trap_robs = transition_robustness(
            transitions, atom_rob_dict
        )

        positive_robs: list[RobustnessCounter] = [
            RobustnessCounter(rob, i)
            for i, rob in enumerate(robs)
            if int(math.copysign(1, rob)) == 1
        ]

        # Check if there is only one positive transition robustness unless there are
        # multiple 0's
        if len(positive_robs) != 1:
            is_all_positive_zero: bool = all(
                int(pos_rob.robustness) == 0 for pos_rob in positive_robs
            )
            if is_all_positive_zero:
                is_containing_trap_state: bool = False
                trap_index = 0
                for i, pos_rob in enumerate(positive_robs):
                    if transitions[pos_rob.ind].is_trapped_next:
                        is_containing_trap_state = True
                        trap_index = i
                        break
                    else:
                        pass

                if is_containing_trap_state:
                    positive_robs = [positive_robs[trap_index]]
                else:
                    next_states: list[AutomatonStateCounter] = []
                    for pos_rob_ind, pos_rob in enumerate(positive_robs):
                        next_states.append(
                            AutomatonStateCounter(
                                transitions[pos_rob.ind].next_state, pos_rob_ind
                            )
                        )
                    next_state_inds: list[int] = [
                        state.ind
                        for state in next_states
                        if state.ind != curr_aut_state
                    ]
                    if next_state_inds:
                        positive_robs = [positive_robs[random.choice(next_state_inds)]]
                    else:  # should only contain the current state as the next state
                        positive_robs = [random.choice(positive_robs)]

            else:
                print("Error: Only one of the transition robustnesses can be positive.")
                print("The positive transitions were:")
                print(
                    [transitions[rob[1]].condition for rob in positive_robs],
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            pass
        positive_rob: RobustnessCounter = deque(positive_robs).pop()
        trans_rob: float = positive_rob.robustness
        next_aut_state: int = transitions[positive_rob.ind].next_state

        # Calculate the reward
        reward: float
        # Weight for reward calculation
        alpha: float = 0.7
        beta: float = 0.5
        gamma: float = 0.0001

        if next_aut_state == curr_aut_state:
            # non_trap_robs.remove(trans_rob)
            # reward = -gamma * (
            #    beta * 1 / max(non_trap_robs) - (1 - beta) * 1 / max(trap_robs)
            # )

            # reward = gamma * (
            #    beta * 1 / max(non_trap_robs) - (1 - beta) * 1 / max(trap_robs)
            # )
            if dense_reward:
                non_trap_robs.remove(trans_rob)
                reward = gamma * max(non_trap_robs)
            else:
                reward = 0
        else:
            if trans_rob in non_trap_robs:
                reward = (
                    10 * trans_rob
                )  # alpha * trans_rob - (1 - alpha) * max(trap_robs)
            elif trans_rob in trap_robs:
                reward = (
                    -10 * trans_rob
                )  # -(1 - alpha) * max(non_trap_robs) - alpha * trans_rob
            else:
                print(
                    "Error: the transition robustness doesn't exit in the robustness set.",
                    file=sys.stderr,
                )
                sys.exit(1)

        return (reward, next_aut_state)


def penalty(trap_robustness: tuple[float], single_penalty: float = -5) -> float:
    """
    Calculate the penalty for getting into the trap state

    Parameters:
         trap_rhobustness (tuple[float]): rhobustness which leads to the trap state q_Tr
    single_penaly (float): penalty for each violating rhobustness

    Returns:
         penaly (float): total penalty
    """
    return np.sum(np.array(trap_robustness) > 0) * single_penalty

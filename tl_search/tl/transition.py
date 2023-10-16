import numpy as np
from numpy.typing import NDArray

from tl_search.tl.reward import transition_robustness

from tl_search.tl.synthesis import TLAutomaton
from tl_search.common.typing import Transition


def find_available_transitions(
    atom_rob_dict: dict[str, float], aut: TLAutomaton
) -> list[Transition]:
    robs: list[list[float]] = []
    atom_prop_names: list[str] = aut.AP.APs

    for edge in aut.edges:
        robs_edge, _, _ = transition_robustness(
            edge.transitions, atom_prop_names, atom_rob_dict
        )
        robs.append(robs_edge)

    pos_rob_inds_np: tuple[NDArray, NDArray] = np.where(np.array(robs) >= 0)
    pos_rob_inds_tmp: list[list[float]] = [inds.tolist() for inds in pos_rob_inds_np]
    pos_rob_inds: list[tuple[int, int]] = list(zip(*pos_rob_inds_tmp))

    transitions: list[Transition] = [
        aut.edges[ind_pair[0]].transitions[ind_pair[1]] for ind_pair in pos_rob_inds
    ]

    return transitions


def find_trap_transitions(atom_rob_dict: dict[str, float], aut: TLAutomaton) -> bool:
    is_trapped: bool = False

    for edge in aut.edges:
        _, _, trap_robs = transition_robustness(edge.transitions, atom_rob_dict)
        if trap_robs:
            for rob in trap_robs:
                if rob >= 0:
                    is_trapped = True
                else:
                    pass
        else:
            pass

    return is_trapped

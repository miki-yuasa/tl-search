import numpy as np
from numpy.typing import NDArray
from scipy.special import entr


def select_max_entropy_target_replicate(
    target_action_probs_list: list[NDArray],
) -> tuple[int, list[float]]:
    max_entropy: float = 0
    max_entropy_idx: int = 0
    entropies: list[float] = []

    for idx, target_action_probs in enumerate(target_action_probs_list):
        target_entropy: float = float(np.sum(entr(target_action_probs)))
        entropies.append(target_entropy)
        if target_entropy > max_entropy:
            max_entropy = target_entropy
            max_entropy_idx = idx

    return max_entropy_idx, entropies


def select_max_entropy_spec_replicate(
    rep_action_probs_list: list[NDArray], rep_trap_masks: list[NDArray]
) -> tuple[int, list[float], list[int]]:
    max_entropy: float = 0
    max_entropy_idx: int = 0
    entropies: list[float] = []

    for idx, (rep_action_probs, rep_trap_mask) in enumerate(
        zip(rep_action_probs_list, rep_trap_masks)
    ):
        rep_entropy: float = float(np.sum(entr(rep_action_probs[rep_trap_mask.astype(np.int32)])))
        entropies.append(rep_entropy)
        if rep_entropy > max_entropy:
            max_entropy = rep_entropy
            max_entropy_idx = idx

    num_non_trap_states: list[int] = []
    for rep_trap_mask in rep_trap_masks:
        num_non_trap_states.append(int(np.sum(rep_trap_mask == 1)))

    return max_entropy_idx, entropies, num_non_trap_states

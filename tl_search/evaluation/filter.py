import numpy as np
from numpy.typing import NDArray

from tl_search.common.typing import FilterMode, PositiveRewardLogger, SpecNode


def filter_rewards(
    reward_threshold: float,
    tl_specs: list[str],
    kl_divs: list[float],
    mean_rewards: list[list[float]],
    model_names: list[str],
    filter_mode: FilterMode = "mean",
) -> PositiveRewardLogger:
    mean_rewards_positive_inds: list[int] = (
        [
            i
            for i, rewards in enumerate(mean_rewards)
            if all(reward >= reward_threshold for reward in rewards)
        ]
        if filter_mode == "elementwise"
        else [
            i
            for i, reward in enumerate(mean_rewards)
            if np.mean(np.array(reward)) >= reward_threshold
        ]
    )

    mean_rewards_pos: list[list[float]] = [
        mean_rewards[i] for i in mean_rewards_positive_inds
    ]
    mean_rewards_pos_out: list[float] = (
        np.mean(np.array(mean_rewards_pos), axis=1).tolist() if mean_rewards_pos else []
    )
    tl_specs_pos: list[str] = [tl_specs[ind] for ind in mean_rewards_positive_inds]
    kl_divs_pos: list[float] = [kl_divs[ind] for ind in mean_rewards_positive_inds]
    model_names_pos: list[str] = (
        [model_names[ind] for ind in mean_rewards_positive_inds] if model_names else []
    )

    # tl_spec: str = tl_specs_pos[int(np.argmin(np.array(kl_divs_pos)))]

    return PositiveRewardLogger(
        tl_specs_pos,
        mean_rewards_pos_out,
        kl_divs_pos,
        mean_rewards_positive_inds,
        model_names_pos,
    )


def apply_filter(
    neighbor_nodes: list[SpecNode],
    neighbor_specs: list[str],
    kl_divs: list[float],
    mean_rewards: list[float] | None = None,
    mean_episode_lengths: list[float] | None = None,
    target_episode_length_mean: float | None = None,
    target_episode_length_std: float | None = None,
    reward_threshold: float = 0.0,
    episode_length_sigma: float | None = 2.16,
):
    out_nodes: list[SpecNode] = neighbor_nodes
    out_specs: list[str] = neighbor_specs
    out_kl_divs: list[float] = kl_divs
    out_mean_rewards: list[float] = []
    out_mean_episode_lengths: list[float] = []

    if mean_rewards is not None:
        reward_means_np: NDArray = np.array(mean_rewards)
        pos_reward_idxs: NDArray = reward_means_np > reward_threshold
        out_mean_rewards = reward_means_np[pos_reward_idxs].tolist()
        out_specs = np.array(out_specs)[pos_reward_idxs].tolist()
        out_nodes = [out_nodes[i] for i, flag in enumerate(pos_reward_idxs) if flag]
        out_kl_divs: list[float] = np.array(kl_divs)[pos_reward_idxs].tolist()

        if mean_episode_lengths is not None:
            out_mean_episode_lengths = np.array(mean_episode_lengths)[
                pos_reward_idxs
            ].tolist()
        else:
            pass
    else:
        pass

    if (
        episode_length_sigma is not None
        and mean_episode_lengths is not None
        and target_episode_length_mean is not None
        and target_episode_length_std is not None
    ):
        mean_episode_lengths_np: NDArray = np.array(out_mean_episode_lengths)
        filtered_episode_idxs: NDArray = np.abs(
            mean_episode_lengths_np - target_episode_length_mean
        ) < (episode_length_sigma * target_episode_length_std)

        out_specs_tmp = np.array(out_specs)[filtered_episode_idxs].tolist()

        out_specs = out_specs_tmp
        out_nodes = [
            out_nodes[i] for i, flag in enumerate(filtered_episode_idxs) if flag
        ]
        out_kl_divs: list[float] = np.array(out_kl_divs)[filtered_episode_idxs].tolist()
        out_mean_episode_lengths = np.array(mean_episode_lengths)[
            pos_reward_idxs
        ].tolist()

        if mean_rewards is not None:
            out_mean_rewards = np.array(out_mean_rewards)[
                filtered_episode_idxs
            ].tolist()
        else:
            pass

    else:
        pass

    return out_nodes, out_specs, out_kl_divs, out_mean_rewards, out_mean_episode_lengths

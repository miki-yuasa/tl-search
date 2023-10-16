import math

import numpy as np
from numpy.typing import NDArray

from tl_search.common.typing import ActionProb


def kl_div(p: NDArray, q: NDArray) -> NDArray:
    kl_div = np.sum(p * np.nan_to_num(np.log(np.nan_to_num(p / q))), axis=1)

    return kl_div


def kl_div_vec(p: NDArray, q: NDArray) -> float:
    kl_div = np.sum(p * np.nan_to_num(np.log(np.nan_to_num(p / q))))

    return kl_div


def calculate_kl_divs(
    ori_probs: list[ActionProb], tar_probs: list[ActionProb]
) -> list[float]:
    kl_divs: list[float] = []
    for ori_prob, tar_prob in zip(ori_probs, tar_probs):
        kl_div: float = (
            float(kl_div_vec(np.array(tar_prob), np.array(ori_prob)))
            if sum(ori_prob) == 0.0
            else float(kl_div_vec(np.array(ori_prob), np.array(tar_prob)))
        )
        kl_divs.append(kl_div)

    return kl_divs


def kl_div_prob(ori_probs: list[ActionProb], tar_probs: list[ActionProb]) -> float:
    return np.mean(np.array(calculate_kl_divs(ori_probs, tar_probs)))


def find_min_kl_div_tl_spec(
    tl_specs: list[str],
    kl_divs: list[float],
    k_smallest: int = 0,
) -> tuple[str, int, float, float, float]:
    kl_div_means: NDArray = np.array(kl_divs)

    max_kl_div_mean: float = float(np.max(kl_div_means))
    mean_kl_div_mean: float = float(np.mean(kl_div_means))
    try:
        min_kl_ind: int = np.argsort(kl_div_means)[k_smallest]
    except:
        min_kl_ind: int = np.argmin(kl_div_means)
    min_kl_div_mean: float = float(kl_div_means[min_kl_ind])
    min_tl_spec: str = tl_specs[min_kl_ind]

    return (min_tl_spec, min_kl_ind, min_kl_div_mean, mean_kl_div_mean, max_kl_div_mean)


def find_max_reward_spec(
    tl_specs: list[str], mean_rewards: list[list[float]]
) -> tuple[str, float, float, float]:
    reward_means: list[float] = [np.mean(reward_list) for reward_list in mean_rewards]

    max_reward_mean: float = float(np.max(reward_means))
    mean_reward_mean: float = float(np.mean(reward_means))
    min_reward_mean: float = float(np.min(reward_means))
    max_reward_spec: str = tl_specs[np.argmax(reward_means)]

    return (max_reward_spec, min_reward_mean, mean_reward_mean, max_reward_mean)


def moving_average(xx: NDArray, size: int) -> NDArray:
    b = np.ones(size) / size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(size / 2)

    # 補正部分
    xx_mean[0] *= size / n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size / (i + n_conv)
        xx_mean[-i] *= size / (i + n_conv - (size % 2))
    # size%2は奇数偶数での違いに対応するため

    return xx_mean


def confidence_interval(x: NDArray, z_star: float) -> tuple[float, float]:
    n: int = x.size
    x_mean: float = np.mean(x)
    unbiased_var = 1 / (n - 1) * np.sum((x - x_mean) ** 2)
    ci: tuple[float, float] = (
        float(x_mean - z_star * np.sqrt(unbiased_var) / n),
        float(x_mean + z_star * np.sqrt(unbiased_var) / n),
    )

    return ci

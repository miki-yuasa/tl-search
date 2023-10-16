from heapq import nlargest, nsmallest, heappop
import heapq

from tl_search.common.typing import ExperimentLogger, RankOption
from tl_search.common.utils import kl_div


def rank_by(data: ExperimentLogger, rank_option: RankOption, num_rank: int):
    tupled_data: list[tuple] = (
        [
            (kl_div, spec, reward)
            for kl_div, spec, reward in zip(
                data.kl_divs, data.tl_specs, data.mean_rewards
            )
        ]
        if rank_option == "kl_div"
        else [
            (reward, spec, kl_div)
            for kl_div, spec, reward in zip(
                data.kl_divs, data.tl_specs, data.mean_rewards
            )
        ]
    )

    ranked_data = (
        nsmallest(num_rank, tupled_data)
        if rank_option == "kl_div"
        else nlargest(num_rank, tupled_data)
    )

    return ranked_data


def sort_spec(specs: list[str], kl_divs: list[float]) -> tuple[list[str], list[float]]:
    couples: list[tuple[float, str]] = [
        (kl_div, spec) for spec, kl_div in zip(specs, kl_divs)
    ]

    heapq.heapify(couples)

    sorted_specs: list[str] = []
    sorted_kl_divs: list[float] = []

    num_itr: int = len(couples)
    for i in range(num_itr):
        couple: tuple[float, str] = heappop(couples)
        kl_div, spec = couple
        sorted_specs.append(spec)
        sorted_kl_divs.append(kl_div)

    return sorted_specs, sorted_kl_divs

import itertools
from typing import Union, cast, overload

import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr
from scipy.stats import rankdata
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from tl_search.tl.environment import restore_model

from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import atom_tl_ob2rob
from tl_search.tl.transition import find_trap_transitions
from tl_search.common.utils import confidence_interval
from tl_search.evaluation.extractor import (
    get_action_distribution,
    get_agent_action_distributions,
    get_agent_actions,
)
from tl_search.common.typing import (
    ActionProb,
    FixedMapLocations,
    KLDivReport,
    KLDivReportDict,
    MapLocations,
    ModelProps,
    ObsProp,
)
from tl_search.map.utils import sample_agent_locs


def evaluate_spec(
    model: PPO,
    map_locs_list: list[MapLocations],
    ori_action_probs_np: list[NDArray],
    is_trap_state_filtered: Union[bool, None],
    is_reward_calculated: bool,
    is_similarity_compared: bool = True,
) -> tuple[list[ActionProb], list[ActionProb], float, float]:
    print("Evaluating model.")

    ori_action_probs_tmp: list[ActionProb] = (
        [
            cast(ActionProb, tuple(ori_action_prob.astype(object)))
            for ori_action_prob in ori_action_probs_np
        ]
        if is_similarity_compared
        else []
    )

    tl_action_probs: list[ActionProb] = []
    ori_action_probs: list[ActionProb] = []

    if is_similarity_compared:
        for map_locs, ori_action_prob in zip(map_locs_list, ori_action_probs_tmp):
            aut: TLAutomaton = model.env.envs[0].env.aut
            atom_rob_dict, obs_dict = atom_tl_ob2rob(aut, map_locs)

            is_trapped: bool = False

            if is_trap_state_filtered:
                is_trapped = find_trap_transitions(atom_rob_dict, aut)
            else:
                pass

            if is_trapped:
                tl_action_probs.append((0.2, 0.2, 0.2, 0.2, 0.2))
            else:
                policy: ActorCriticPolicy = cast(ActorCriticPolicy, model.policy)
                tl_action_prob: NDArray = get_action_distribution(
                    policy, torch.tensor([list(obs_dict.values())]).to(model.device)
                )
                tl_action_probs.append(
                    cast(ActionProb, tuple(tl_action_prob.astype(object)))
                )
                ori_action_probs.append(ori_action_prob)
    else:
        pass

    if is_reward_calculated:
        mean_reward, std_reward = evaluate_policy(
            model, model.get_env(), n_eval_episodes=100
        )
    else:
        mean_reward = None
        std_reward = None

    return (
        tl_action_probs,
        ori_action_probs,
        cast(float, mean_reward),
        cast(float, std_reward),
    )


def compare_ppo_models(
    reference: PPO,
    learned: PPO,
    num_samples: int,
    fixed_map_locs: FixedMapLocations,
    is_from_all_territories: bool,
) -> tuple[NDArray, float, float, float, float, tuple[float, float]]:
    map_locs_list: list[MapLocations] = [
        sample_agent_locs(fixed_map_locs, is_from_all_territories)
        for _ in range(num_samples)
    ]

    ref_action_probs, ref_trap_mask = get_agent_action_distributions(
        reference, map_locs_list
    )
    learned_action_probs, learned_trap_mask = get_agent_action_distributions(
        learned, map_locs_list
    )

    trap_mask: NDArray = ref_trap_mask * learned_trap_mask

    ref_action_probs_filtered: NDArray = ref_action_probs[trap_mask == 1]
    learned_action_probs_filtered: NDArray = learned_action_probs[trap_mask == 1]

    kl_divs: NDArray = np.sum(
        rel_entr(ref_action_probs_filtered, learned_action_probs_filtered), axis=1
    )

    kl_div_mean, kl_div_std, kl_div_max, kl_div_min, kl_ci = collect_kl_div_stats(
        kl_divs
    )

    return kl_divs, kl_div_mean, kl_div_std, kl_div_max, kl_div_min, kl_ci


def compare_dqn_models(
    reference: DQN,
    learned: DQN,
    num_samples: int,
    fixed_map_locs: FixedMapLocations,
    is_from_all_territories: bool = False,
) -> tuple[int, int, float]:
    map_locs_list: list[MapLocations] = [
        sample_agent_locs(fixed_map_locs, is_from_all_territories)
        for _ in range(num_samples)
    ]

    ref_actions, ref_trap_mask = get_agent_actions(reference, map_locs_list)
    learned_actions, learned_trap_mask = get_agent_actions(learned, map_locs_list)

    trap_mask: NDArray = ref_trap_mask * learned_trap_mask

    ref_actions_filtered: NDArray = ref_actions[trap_mask == 1]
    learned_actions_filtered: NDArray = learned_actions[trap_mask == 1]

    comparison: NDArray = ref_actions_filtered == learned_actions_filtered

    num_matched: int = np.sum(comparison)
    num_elements: int = comparison.size

    accuracy: float = num_matched / num_elements

    return num_elements, num_matched, accuracy


@overload
def collect_kl_div_stats(kl_divs: NDArray, in_dict: bool = True) -> KLDivReportDict:
    ...


@overload
def collect_kl_div_stats(kl_divs: NDArray, in_dict: bool = False) -> KLDivReport:
    ...


def collect_kl_div_stats(kl_divs: NDArray, in_dict: bool = False):
    kl_div_mean: float = float(np.mean(kl_divs))
    kl_div_std: float = float(np.std(kl_divs))
    kl_div_max: float = float(np.max(kl_divs))
    kl_div_min: float = float(np.min(kl_divs))
    kl_div_median: float = float(np.median(kl_divs))

    try:
        kl_ci: tuple[float, float] = confidence_interval(kl_divs, 1.96)
    except:
        kl_ci = (0, 0)

    report: KLDivReport | KLDivReportDict
    if in_dict:
        report = {
            "kl_div_mean": kl_div_mean,
            "kl_div_std": kl_div_std,
            "kl_div_max": kl_div_max,
            "kl_div_min": kl_div_min,
            "kl_div_median": kl_div_median,
            "kl_ci": kl_ci,
        }
    else:
        report = KLDivReport(kl_div_mean, kl_div_std, kl_div_max, kl_div_min, kl_ci)
    return report


def evaluate_models(
    ref_model: PPO,
    lerarned_model_props_list: list[ModelProps],
    num_samples: int,
    is_initialized_in_both_territories: bool,
):
    kl_divs_all: list[NDArray] = []
    kl_div_reports: list[KLDivReport] = []

    for model_props in lerarned_model_props_list:
        learned_model: PPO = restore_model(
            model_props, ref_model.device, is_initialized_in_both_territories
        )

        (
            kl_divs,
            kl_div_mean,
            kl_div_std,
            kl_div_max,
            kl_div_min,
            kl_ci,
        ) = compare_ppo_models(
            ref_model,
            learned_model,
            num_samples,
            model_props.map_props.fixed_map_locs,
        )

        kl_divs_all.append(kl_divs)
        kl_div_reports.append(
            KLDivReport(kl_div_mean, kl_div_std, kl_div_max, kl_div_min, kl_ci)
        )

    kl_div_report_all = KLDivReport(
        *collect_kl_div_stats(np.array(kl_divs_all).ravel())
    )

    return kl_divs_all, kl_div_reports, kl_div_report_all


def evaluate_ppo_models(
    ind_combs: list[tuple[int, int]],
    ref_action_probs_list: list[NDArray],
    ref_trap_masks: list[NDArray],
    learned_models_props: list[ModelProps],
    obs_props: list[ObsProp],
    map_locs_list: list[MapLocations],
    device: torch.device,
    is_initialized_in_both_territories: bool,
) -> tuple[KLDivReport, list[KLDivReport], NDArray, float, float]:
    print("Evaluating " + learned_models_props[0].spec)
    learned_action_probs_list: list[NDArray] = []
    learned_trap_masks: list[NDArray] = []

    rewards_model: list[float] = []

    for props in learned_models_props:
        print("Simulating model {}".format(props.model_name))
        model = cast(
            PPO,
            restore_model(props, obs_props, device, is_initialized_in_both_territories),
        )
        learned_action_probs, learned_trap_mask = get_agent_action_distributions(
            model, map_locs_list
        )
        learned_action_probs_list.append(learned_action_probs)
        learned_trap_masks.append(learned_trap_mask)

        rewards, _ = evaluate_policy(
            model, model.get_env(), n_eval_episodes=100, return_episode_rewards=True
        )

        rewards_model += cast(list[float], rewards)

    rewards_model_np = np.array(rewards_model)
    reward_mean: float = np.mean(rewards_model_np)
    reward_std: float = np.std(rewards_model_np)

    kl_divs_model: list[NDArray] = []
    kl_div_reports: list[KLDivReport] = []

    for comb in ind_combs:
        ref_ind, learned_ind = comb

        trap_mask: NDArray = ref_trap_masks[ref_ind] * learned_trap_masks[learned_ind]

        ref_action_probs_filtered: NDArray = ref_action_probs_list[ref_ind][
            trap_mask == 1
        ]
        learned_action_probs_filtered: NDArray = learned_action_probs_list[learned_ind][
            trap_mask == 1
        ]

        kl_divs: NDArray = np.sum(
            rel_entr(ref_action_probs_filtered, learned_action_probs_filtered),
            axis=1,
        )

        kl_div_report = (
            KLDivReport(999, 0, 999, 999, (0, 0))
            if kl_divs.size == 0
            else KLDivReport(*collect_kl_div_stats(kl_divs))
        )

        kl_divs_model.append(kl_divs.flatten())
        kl_div_reports.append(kl_div_report)

    kl_divs_concat = np.concatenate(kl_divs_model)
    kl_div_report_model = (
        KLDivReport(999, 0, 999, 999, (0, 0))
        if kl_divs_concat.size == 0
        else KLDivReport(*collect_kl_div_stats(kl_divs_concat))
    )

    kl_div_means: list[float] = [report.kl_div_mean for report in kl_div_reports]

    kl_div_mean_rank = rankdata(kl_div_means)

    return (
        kl_div_report_model,
        kl_div_reports,
        kl_div_mean_rank,
        reward_mean,
        reward_std,
    )

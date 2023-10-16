import itertools

from stable_baselines3 import PPO, DQN
import torch
from tl_search.evaluation.evaluation import compare_dqn_models, compare_ppo_models

from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.tl.environment import restore_model
from tl_search.common.typing import MapProps, EnemyPolicyMode, ModelProps, RLAlgorithm
from tl_search.map.utils import map2props
from tl_search.tl.tl_parser import get_used_tl_props

num_samples: int = 8_000

reference_spec: str = "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"
learned_specs: list[str] = [
    # "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"
    "F(!psi_ba_ra&psi_ba_rf) & G(psi_ba_bf|!psi_ba_obs)",
    "F(psi_ba_rf) & G(psi_ba_ra|psi_ba_bf|!psi_ba_obs)",
    "F(!psi_ba_bf&psi_ba_rf) & G(psi_ba_ra|!psi_ba_obs)",
]

enemy_policy: EnemyPolicyMode = "random"
rl_algorithm: RLAlgorithm = "ppo"
reference_model_name: str = "policies/learned/learned_policy_search_random_12_0_0"  # "models/single/policy_random_dqn_15_0_0"
is_from_all_territories: bool = True

learned_model_names_list: list[list[str]] = [
    # [
    #     "models/single/learned_policy_search_random_13_0_0",
    #     "models/single/learned_policy_search_random_13_0_1",
    #     "models/single/learned_policy_search_random_13_0_2",
    # ]
    # [
    #     "models/single/policy_random_dqn_15_0_0",
    #     "models/single/policy_random_dqn_15_0_1",
    #     "models/single/policy_random_dqn_15_0_2",
    # ],
    [
        "models/single/learned_policy_search_random_14_0_0",
        "models/single/learned_policy_search_random_14_0_1",
        "models/single/learned_policy_search_random_14_0_2",
    ],
    [
        "models/single/learned_policy_search_random_14_1_0",
        "models/single/learned_policy_search_random_14_1_1",
        "models/single/learned_policy_search_random_14_1_2",
    ],
    [
        "models/single/learned_policy_search_random_14_2_0",
        "models/single/learned_policy_search_random_14_2_1",
        "models/single/learned_policy_search_random_14_2_2",
    ],
]

map_filename: str = "maps/board_0002.txt"

model_labels: list[str] = ["reference", "learned_1", "learned_2", "learned_3"]

gpu: int = 1

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))

reference_model_props = ModelProps(
    reference_model_name,
    rl_algorithm,
    reference_spec,
    *get_used_tl_props(reference_spec, atom_prop_dict_all, obs_props_all),
    enemy_policy,
    map_props,
)
device = torch.device("cuda", gpu)

reference_model: PPO | DQN = restore_model(
    reference_model_props,
    device,
    True,
)

for learned_spec, learned_model_names in zip(learned_specs, learned_model_names_list):
    print("Evaluating spec: " + learned_spec)
    models: list[PPO | DQN] = [reference_model] + [
        restore_model(
            ModelProps(
                model_name,
                rl_algorithm,
                learned_spec,
                *get_used_tl_props(learned_spec, atom_prop_dict_all, obs_props_all),
                enemy_policy,
                map_props,
            ),
            device,
            True,
        )
        for model_name in learned_model_names
    ]

    model_combinations: list[tuple[int, int]] = list(
        itertools.combinations(range(len(learned_model_names) + 1), 2)
    )

    for comb in model_combinations:
        model_1 = models[comb[0]]
        model_2 = models[comb[1]]
        if isinstance(model_1, PPO) and isinstance(model_2, PPO):
            (
                kl_divs,
                kl_div_mean,
                kl_div_std,
                kl_div_max,
                kl_div_min,
                kl_ci,
            ) = compare_ppo_models(
                model_1,
                model_2,
                num_samples,
                map_props.fixed_map_locs,
                is_from_all_territories,
            )

            print(
                "Comparing models: {} & {}".format(
                    model_labels[comb[0]], model_labels[comb[1]]
                )
            )
            print("KL div. mean: {:.6f}".format(kl_div_mean))
            print("KL div. std: {:.6f}".format(kl_div_std))
            print("KL div. max: {:.6f}".format(kl_div_max))
            print("KL div. min: {:.6f}".format(kl_div_min))
            print("KL div. confidence interval: [{:.6f},{:.6f}]".format(*kl_ci))

        elif isinstance(model_1, DQN) and isinstance(model_2, DQN):
            (num_elements, num_matched, accuracy) = compare_dqn_models(
                model_1,
                model_2,
                num_samples,
                map_props.fixed_map_locs,
                is_from_all_territories,
            )

            print(
                "Comparing models: {} & {}".format(
                    model_labels[comb[0]], model_labels[comb[1]]
                )
            )
            print("Number of collected samples: {:.0f}".format(num_elements))
            print("Number of matched actions: {:.0f}".format(num_matched))
            print("Accuracy: {:.3f}".format(accuracy))

        else:
            raise Exception

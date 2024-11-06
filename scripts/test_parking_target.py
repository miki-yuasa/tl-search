import os
import pickle
import json
from typing import Any, Final
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import SAC

from tl_search.common.typing import EntropyReportDict, ObsProp
from tl_search.envs.tl_parking import TLAdversarialParkingEnv
from tl_search.evaluation.eval import collect_kl_div_stats
from tl_search.search.sample import sample_obs
from tl_search.search.search_parking import (
    evaluate_policy,
    gaussian_dist_entropy,
    gaussian_kl_div,
    get_action_distributions,
    return_input,
    select_max_entropy_spec_replicate,
)


num_replicates: Final[int] = 1
num_episodes: Final[int] = 100
num_samples: int = 5000
target_spec: str | None = "F(psi_ego_goal) & G(!psi_ego_adv & !psi_ego_wall)"
target_model_path: str = (
    f"out/models/search/parking/sac_F(psi_ego_goal)_and_G(!psi_ego_adv_and_!psi_ego_wall).zip"
)
kl_div_suffix: str | None = "parking_exp1"

model_path: str = (
    "out/models/parking/parking_demo_fixed_sac_512_512_512_timesteps_0.1M_tl_F(psi_ego_goal)_and_G(!psi_ego_adv_and_!psi_ego_wall)_scaled_1.zip"  # "out/models/search/parking/sac_F(psi_ego_goal)_and_G(!psi_ego_adv_and_!psi_ego_wall)_0.zip"
)


log_suffix: str = (
    f"{kl_div_suffix}_extended_" if kl_div_suffix is not None else "extended_"
)

target_actions_path: str = (
    f"out/data/search/dataset/action_probs_sac_{log_suffix}full.npz"
)
target_entropy_path: str = target_actions_path.replace(".npz", "_ent.json")

obs_list_path: str = f"out/data/search/dataset/obs_list_sac_full.pkl"

config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],  # "heading"],
        "scales": [1, 1, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "reward_weights": [1, 0.2, 0, 0, 0.1, 0.1],
    "success_goal_reward": 0.05,
    "collision_reward": -5,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 50,
    "screen_width": 600,
    "screen_height": 300,
    "screen_center": "centering_position",
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "vehicles_count": 0,
    "adversarial_vehicle": True,
    "add_walls": True,
    "adversarial_vehicle_spawn_config": [
        {"spawn_point": [-30, 4], "heading": 0, "speed": 5},
        {"spawn_point": [-30, -4], "heading": 0, "speed": 5},
        {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
        {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
    ],
    "dense_reward": True,
}

obs_props: list[ObsProp] = [
    ObsProp("d_ego_goal", ["d_ego_goal"], return_input),
    ObsProp("d_ego_adv", ["d_ego_adv"], return_input),
    ObsProp("d_ego_wall", ["d_ego_wall"], return_input),
]

atom_pred_dict: dict[str, str] = {
    "psi_ego_goal": "d_ego_goal < {}".format(2),
    "psi_ego_adv": "d_ego_adv < {}".format(4),
    "psi_ego_wall": "d_ego_wall < {}".format(5),
}

obs_list: list[dict[str, Any]]

sample_env = TLAdversarialParkingEnv(target_spec, config, obs_props, atom_pred_dict)

if os.path.exists(obs_list_path):
    with open(obs_list_path, "rb") as f:
        obs_list = pickle.load(f)
else:
    model = SAC.load(target_model_path, sample_env)
    obs_list = sample_obs(sample_env, model, num_samples)

    # Save the observations
    with open(obs_list_path, "wb") as f:
        pickle.dump(obs_list, f)
target_gaus_means_list: list[NDArray]
target_gaus_stds_list: list[NDArray]
target_trap_masks: list[NDArray]
if os.path.exists(target_actions_path):
    print("Loading target actions...")
    npz = np.load(target_actions_path)
    target_gaus_means_list = npz["gaus_means"]
    target_gaus_stds_list = npz["gaus_stds"]
    target_trap_masks = npz["masks"]
else:
    print("Generating target actions...")
    target_env = TLAdversarialParkingEnv(target_spec, config, obs_props, atom_pred_dict)

    target_gaus_means_list = []
    target_gaus_stds_list = []
    target_trap_masks = []
    print("Loading target model...")
    for i in range(num_replicates):
        print(f"Replicate {i}")
        model = SAC.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)
        action_probs: NDArray
        gaus_means, gaus_stds, trap_mask = get_action_distributions(
            model, target_env, obs_list
        )
        target_gaus_means_list.append(gaus_means)
        target_gaus_stds_list.append(gaus_stds)
        target_trap_masks.append(trap_mask)

        del model

    np.savez(
        target_actions_path,
        gaus_means=target_gaus_means_list,
        gaus_stds=target_gaus_stds_list,
        masks=target_trap_masks,
    )

if os.path.exists(target_entropy_path):
    with open(target_entropy_path, "r") as f:
        target_entropy_dict: EntropyReportDict = json.load(f)

    target_max_entropy_idx: int = target_entropy_dict["max_entropy_idx"]
else:
    (
        target_max_entropy_idx,
        target_entropies,
        num_non_trap_states,
    ) = select_max_entropy_spec_replicate(target_gaus_stds_list, target_trap_masks)

    target_entropy_dict: EntropyReportDict = {
        "max_entropy_idx": target_max_entropy_idx,
        "entropies": target_entropies,
        "num_non_trap_states": num_non_trap_states,
    }

    with open(target_entropy_path, "w") as f:
        json.dump(target_entropy_dict, f, indent=4)

target_gaus_means_list = [target_gaus_means_list[target_max_entropy_idx]]
target_gaus_stds_list = [target_gaus_stds_list[target_max_entropy_idx]]
target_trap_masks = [target_trap_masks[target_max_entropy_idx]]

model_kl_divs: list[NDArray] = []

env = TLAdversarialParkingEnv(target_spec, config, obs_props, atom_pred_dict)

rep_gaus_means_list: list[NDArray[np.float64]] = []
rep_gaus_stds_list: list[NDArray[np.float64]] = []
rep_trap_masks: list[NDArray[np.float64]] = []

spec_models = [SAC.load(model_path, env)]

# model_rewards: list[float] = []
# episode_lengths: list[int] = []

# for i, model in enumerate(spec_models):
#     rewards: list[float]
#     rewards, lengths = evaluate_policy(model, env, num_episodes)

#     model_rewards += rewards
#     episode_lengths += lengths

# print(f"Model rewards: {np.mean(model_rewards)}")
# print(f"Episode lengths: {np.mean(episode_lengths)}")

for i, model in enumerate(spec_models):
    print(
        f"Getting action distributions for {env._tl_spec} for rep {i + 1}/{len(spec_models)}..."
    )

    rep_gaus_means, rep_gaus_stds, rep_trap_mask = get_action_distributions(
        model, env, obs_list
    )
    rep_gaus_means_list.append(rep_gaus_means)
    rep_gaus_stds_list.append(rep_gaus_stds)
    rep_trap_masks.append(rep_trap_mask)

(
    max_entropy_idx,
    entropies,
    num_non_trap_states,
) = select_max_entropy_spec_replicate(rep_gaus_stds_list, rep_trap_masks)

print(f"Getting KL divergences for {env._tl_spec}...")

max_kl_div: float = gaussian_dist_entropy(10)

# for comb in idx_combs:
target_idx = 0
spec_idx = 0

print(f"Calculating KL divergence for {env._tl_spec}...")
trap_mask: NDArray = target_trap_masks[target_idx] * rep_trap_masks[spec_idx]
target_gaus_means_filtered: NDArray = target_gaus_means_list[target_idx][trap_mask == 1]
target_gaus_stds_filtered: NDArray = target_gaus_stds_list[target_idx][trap_mask == 1]
spec_gaus_means_filtered: NDArray = rep_gaus_means_list[spec_idx][trap_mask == 1]
spec_gaus_stds_filtered: NDArray = rep_gaus_stds_list[spec_idx][trap_mask == 1]

print(f"Calculating entropy for {env._tl_spec}...")
target_entropy: NDArray = np.mean(
    gaussian_dist_entropy(target_gaus_stds_filtered), axis=1
)
normalized_entropy: NDArray = 1 - target_entropy / max_kl_div
weight: NDArray = normalized_entropy / np.sum(normalized_entropy)

print(f"Calculating weighted KL divergence for {env._tl_spec}...")
kl_divs: NDArray = (
    np.mean(
        gaussian_kl_div(
            target_gaus_means_filtered,
            target_gaus_stds_filtered,
            spec_gaus_means_filtered,
            spec_gaus_stds_filtered,
        ),
        axis=1,
    )
    * weight
)

model_kl_divs.append(kl_divs.flatten())

model_kl_divs_concat: NDArray = np.concatenate(model_kl_divs)

print(collect_kl_div_stats(model_kl_divs_concat, in_dict=True))

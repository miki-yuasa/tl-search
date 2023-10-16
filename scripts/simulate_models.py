from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tl_search.tl.constants import (
    known_policy,
    obs_props_all,
    atom_prop_dict_all,
)
from tl_search.tl.environment import create_env
from tl_search.evaluation.visualize import create_animation, simulate_model
from tl_search.common.typing import EnemyPolicyMode, PolicyMode

policy_mode: PolicyMode = "known"
model_suffix: str = "models/single/policy_random_ppo_17_0"
num_replicates: int = 3
spec: str = "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"  # "F (psi_ba_ra & !psi_ba_rt) & G( (psi_ra_bf->psi_ra_bt) & !psi_ba_obwa)"
enemy_policy: EnemyPolicyMode = "random"
is_initialized_both_territories: bool = False
run: int = 17

model_names: list[str] = [model_suffix + "_{}".format(i) for i in range(num_replicates)]
# model_names: list[str] = [model_suffix]

for i, model_name in enumerate(model_names):
    env = create_env(
        spec,
        known_policy,
        obs_props_all,
        atom_prop_dict_all,
        enemy_policy,
        is_initialized_both_territories,
    )
    env_monitored = Monitor(env)
    model = PPO.load(model_name, env)

    env_done = simulate_model(model, env)
    video_save_name: str = f"out/plots/animation/single/policy_random_{run}_0_{i}.gif"
    create_animation(
        env_done.blue_path,
        env_done.red_path,
        env_done.init_map,
        env_done.map_object_ids,
        video_save_name,
        False,
    )

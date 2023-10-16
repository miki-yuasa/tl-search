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
model_name: str = "models/single/policy_random_ppo_17_0_0"
num_replicates: int = 1
spec: str = "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"
num_games: int = 1000

enemy_policy: EnemyPolicyMode = "random"
is_initialized_both_territories: bool = False

# model_names: list[str] = [model_suffix + "_{}".format(i) for i in range(num_replicates)]

env = create_env(
    spec,
    known_policy,
    obs_props_all,
    atom_prop_dict_all,
    enemy_policy,
    is_initialized_both_territories,
)
env_monitored = Monitor(env)
model = PPO.load(model_name, env_monitored)

win_counter: int = 0

for _ in range(num_games):
    env_done = simulate_model(model, env)
    if env_done.aut_state in env_done.aut.goal_states:
        win_counter += 1
    else:
        pass

print(win_counter)

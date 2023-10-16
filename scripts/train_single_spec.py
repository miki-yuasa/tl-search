import pickle

from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.map.utils import map2props
from train.train import train_spec
from tl_search.common.seed import torch_fix_seed
from tl_search.common.typing import (
    EnemyPolicyMode,
    EnvProps,
    ModelProps,
    PolicyMode,
    RLAlgorithm,
    SaveMode,
    MapProps,
)
from tl_search.tl.tl_parser import get_used_tl_props

policy_mode: PolicyMode = "known"
rl_algorithm: RLAlgorithm = "ppo"
enemy_policy: EnemyPolicyMode = "random"
map_filename: str = "maps/board_0002.txt"

num_replicates: int = 3
total_timesteps: int = 500_000  # 750_000  # 500_000
seed: float = 3406
replicate_seeds: list[int] = [406 + i * 1000 for i in range(num_replicates)]

gpu: int = 1

run: int = 0


specs: list[str] = [
    "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)",
    # "F(!psi_ba_ra&psi_ba_rf) & G(psi_ba_bf|!psi_ba_obs)",
    # "F(psi_ba_rf) & G(psi_ba_ra|psi_ba_bf|!psi_ba_obs)",
    # "F(!psi_ba_bf&psi_ba_rf) & G(psi_ba_ra|!psi_ba_obs)",
]
description: str = "debug"
log_dir: str = "./tmp/gym/{}/learned/".format(policy_mode)
suffix: str = "policy_{}_{}_{}".format(enemy_policy, rl_algorithm, run)
model_savename: str = "models/single/{}".format(suffix)
data_savename: str = "out/data/single/{}_data.pickle".format(suffix)
reward_curve_savename: str = "learned/{}".format(suffix)
reward_curve_plot_mode: SaveMode = "enabled"
animation_mode: SaveMode = "enabled"
animation_savename: str = "out/plots/animation/{}".format(suffix)

is_initialized_in_both_territories: bool = True
is_gui_shown: bool = False

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))

torch_fix_seed(seed)

model_props_list: list[list[ModelProps]] = []

for i, spec in enumerate(specs):
    env_props = EnvProps(
        spec,
        *get_used_tl_props(spec, atom_prop_dict_all, obs_props_all),
        enemy_policy,
        map_props,
    )

    model_props = train_spec(
        env_props,
        rl_algorithm,
        i,
        gpu,
        num_replicates,
        total_timesteps,
        log_dir,
        model_savename,
        reward_curve_plot_mode,
        reward_curve_savename,
        is_initialized_in_both_territories,
        animation_mode,
        animation_savename,
        is_gui_shown,
        replicate_seeds,
    )

with open(data_savename, mode="wb") as f:
    pickle.dump(model_props_list, f)

#    kl_div_all += kl_div
# file_suffix: str = "{}_single_{}".format(policy_mode, experiment)


# results = ExperimentLogger(
#    policy_mode,
#    experiment,
#    description,
#    "",
#    "",
#    0.0,
#    0.0,
#    0.0,
#    0.0,
#    0.0,
#    0.0,
#    specs,
#    kl_div_all,
#    [],
#    [],
#    [],
#    [],
#    [],
# )
# save_name: str = "data/single/debug_{}.json".format(experiment)
#
# data_saver(results, save_name)

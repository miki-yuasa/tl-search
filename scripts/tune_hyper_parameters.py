from typing import Union, cast

from tl_search.common.typing import (
    HyperparameterTuningLogger,
    Hyperparameters,
    PolicyMode,
    SaveMode,
)
from train.wrapper import train_evaluate_single_spec
from tl_search.tl.constants import (
    known_policy,
    unknown_policy,
    atom_prop_dict_all,
    obs_props_all,
)
from tl_search.common.tuning import replace_hyperparam
from tl_search.common.io import data_saver

target: str = "max_grad_norm"

spec: str = "F (psi_ba_ra & !psi_ba_rt) & G( (psi_ra_bf->psi_ra_bt) & !psi_ba_obwa)"
policy_mode: PolicyMode = "known"
num_sampling: int = 2000
num_replicates: int = 5
total_timesteps: int = 75_000
animation_mode: SaveMode = "enabled"

learning_rates: list[float] = [1e-5, 3e-4, 5e-4, 1e-4, 5e-5]
n_step_batch_pairs: list[tuple[int, int]] = [
    (2048, 64),
    (2048, 128),
    (1024, 128),
    (2048, 64),
    (1024, 64),
]
n_epochs: list[int] = [10, 5, 1]
gammas: list[float] = [0.99, 0.98, 0.95]
gae_lambdas: list[float] = [0.95, 0.8, 0.6]
clip_ranges: list[float] = [0.2, 0.1, 0.3]
ent_coefs: list[float] = [0.0, 1e-4, 1e-3, 1e-2]
vf_coefs: list[float] = [1.0, 0.5, 0.1]
max_grad_norms: list[float] = [
    0.75,
    0.25,
    0.5,
]


hyperparam_value_table: dict[
    str, Union[list[float], list[int], list[tuple[int, int]]]
] = {
    "learning_rate": learning_rates,
    "n_step_batch_pair": n_step_batch_pairs,
    "n_epochs": n_epochs,
    "gamma": gammas,
    "gae_lambda": gae_lambdas,
    "clip_range": clip_ranges,
    "ent_coef": ent_coefs,
    "vf_coef": vf_coefs,
    "max_grad_norm": max_grad_norms,
}
default_value_table: dict[
    str,
    Union[
        float, int, bool, tuple[int, int], list[float], list[int], list[tuple[int, int]]
    ],
] = {key: values[0] for key, values in hyperparam_value_table.items()}
hyperparam_inputs: list[Hyperparameters] = [
    replace_hyperparam(target, value, default_value_table)
    for value in hyperparam_value_table[target]
]

kl_divs_all: list[list[float]] = []

for i, hyperparams in enumerate(hyperparam_inputs):
    animation_savename: str = "hyperparam_tuning_{}_{}".format(
        target, hyperparam_value_table[target][i]
    )

    kl_divs: list[list[float]] = train_evaluate_single_spec(
        spec,
        policy_mode,
        9,
        "",
        0,
        known_policy,
        unknown_policy,
        num_sampling,
        num_replicates,
        total_timesteps,
        atom_prop_dict_all,
        obs_props_all,
        animation_mode,
        animation_savename,
        False,
        False,
        learning_rate=hyperparams.learning_rate,
        n_steps=hyperparams.n_steps,
        batch_size=hyperparams.batch_size,
        n_epochs=hyperparams.n_epochs,
        gamma=hyperparams.gamma,
        gae_lambda=hyperparams.gae_lambda,
        clip_range=hyperparams.clip_range,
        ent_coef=hyperparams.ent_coef,
        vf_coef=hyperparams.vf_coef,
        max_grad_norm=hyperparams.max_grad_norm,
    )

    kl_divs_all += kl_divs

n_steps_log: Union[int, list[int]]
batch_size_log: Union[int, list[int]]

if target == "n_step_batch_pair":
    n_steps: list[int] = []
    batch_sizes: list[int] = []
    for pair in cast(
        list[tuple[int, int]], hyperparam_value_table["n_step_batch_pair"]
    ):
        n_steps.append(pair[0])
        batch_sizes.append(pair[0])
    n_steps_log = n_steps
    batch_size_log = batch_sizes
else:
    n_steps_log, batch_size_log = cast(
        tuple[int, int], default_value_table["n_step_batch_pair"]
    )

default_value_table[target] = hyperparam_value_table[target]

results = HyperparameterTuningLogger(
    spec,
    default_value_table["learning_rate"],
    n_steps_log,
    batch_size_log,
    default_value_table["n_epochs"],
    default_value_table["gamma"],
    default_value_table["gae_lambda"],
    default_value_table["clip_range"],
    default_value_table["ent_coef"],
    default_value_table["vf_coef"],
    default_value_table["max_grad_norm"],
    kl_divs_all,
)

save_name: str = "data/single/hyperparameter_{}.json".format(target)

data_saver(results, save_name)

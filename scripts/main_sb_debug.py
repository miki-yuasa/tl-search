import argparse

from train.wrapper import train_evaluate_batched_specs
from tl_search.tl.constants import known_policy, unknown_policy, obs_props_all

fight_range: float = 2.0
defense_range: float = 1.0
atom_prop_dict_all: dict[str, str] = {
    "psi_ba_ra": "d_ba_ra < {}".format(fight_range),
    # "psi_ba_rf": "d_ba_rf < 0.5",
    "psi_ba_rt": "d_ba_rt < 0.5",
    "psi_ra_bf": "d_ra_bf < {}".format(defense_range),
    "psi_ra_bt": "d_ra_bt < 0.5",
    "psi_ba_ob": "d_ba_ob < 0.5",
    "psi_ba_wa": "d_ba_wa < 0.5",
}

phi_task: str = "(psi_ba_ra & !psi_ba_rt)"

description: str = "phi_task is now psi_ba_ra & !psi_ba_rt"

train_evaluate_batched_specs(
    "known",
    1,
    8,
    description,
    6,
    16,
    False,
    known_policy,
    unknown_policy,
    2_000,
    3,
    3,
    50_000,
    phi_task,
    atom_prop_dict_all,
    obs_props_all,
    reward_curve_plot_mode="disabled",
)

import argparse

from train.wrapper import train_evaluate_batched_specs
from tl_search.tl.constants import known_policy, unknown_policy, obs_props_all

parser = argparse.ArgumentParser(description="Run RLTL code.")
parser.add_argument(
    "policy_mode", type=str, choices=["known", "unknown"], help="policy (known/unknown)"
)
parser.add_argument("experiment", type=int, help="suffix for the saved data")
parser.add_argument("description", type=str, help="description of the experiment")
parser.add_argument("cuda", type=int, help="GPU to use i.e. cuda:0, cuda:1")
parser.add_argument("process", type=int, help="process number i.e. 1, 2, 3, ...")
parser.add_argument(
    "total_processes", type=int, help="total num of processes", default=2
)
parser.add_argument(
    "num_sampling", type=int, help="num of samples from the target policy", default=2000
)
parser.add_argument(
    "num_replicates", type=int, help="num of replicates per spec", default=5
)
parser.add_argument(
    "total_timesteps", type=int, help="total timesteps to learn a spec", default=50000
)
parser.add_argument("phi_task", type=str, help="set the task spec")
parser.add_argument(
    "write_per_specs", type=int, help="save the reslul every n steps", default=3
)
parser.add_argument(
    "-reward_curve_plot_mode",
    type=str,
    choices=["enabled", "disabled", "suppressed"],
    help="set the reward curve plot mode",
    default="disabled",
)
parser.add_argument(
    "-continue_from_existing_data",
    type=bool,
    help="continue from existing data",
    default=False,
)

args = parser.parse_args()

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
    # "psi_ba_obwa": "(d_ba_wa < 0.5)|(d_ba_ob < 0.5)",
}

phi_task: str = "(psi_ba_ra & !psi_ba_rt)"

train_evaluate_batched_specs(
    args.policy_mode,
    args.cuda,
    args.experiment,
    args.description,
    args.process,
    args.total_processes,
    args.continue_from_existing_data,
    known_policy,
    unknown_policy,
    args.num_sampling,
    args.num_replicates,
    args.total_timesteps,
    args.phi_task,
    atom_prop_dict_all,
    obs_props_all,
    args.write_per_specs,
    reward_curve_plot_mode=args.reward_curve_plot_mode,
)

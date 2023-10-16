from tl_search.tl.tl_parser import is_var, tokenize
from tl_search.tl.constants import known_policy, obs_props_all
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.neighbor import nodes2specs
from tl_search.search.search import search_single_start
from tl_search.common.seed import torch_fix_seed
from tl_search.common.typing import SpecNode, SortedLogger
from tl_search.common.io import data_saver

gpus: tuple[int, ...] = (0, 0)
num_process: int = 8

experiment: int = 9
description: str = "search test"
num_sampling: int = 8000
num_replicates: int = 4
total_timesteps: int = 30000
seed: float = 3406

search_depth: int = 10000

fight_range: float = 2.0
defense_range: float = 1.0
atom_prop_dict_all: dict[str, str] = {
    "psi_ba_ra": "d_ba_ra < {}".format(fight_range),
    # "psi_ba_rf": "d_ba_rf < 0.5",
    "psi_ba_rt": "d_ba_rt < 0.5",
    "psi_ra_bf": "d_ra_bf < {}".format(defense_range),
    # "psi_ra_bt": "d_ra_bt < 0.5",
    "psi_ba_ob": "d_ba_ob < 0.5",
    "psi_ba_wa": "d_ba_wa < 0.5",
    # "psi_ba_obwa": "(d_ba_wa < 0.5)|(d_ba_ob < 0.5)",
    # "psi_ba_uh": "d_ba_uh < 0.5",
    # "psi_ba_lh": "d_ba_lh < 0.5",
    # "psi_ra_uh": "d_ra_uh < 0.5",
    # "psi_ra_lh": "d_ra_lh < 0.5",
}

phi_task: str = "psi_ba_ra & !psi_ba_rt"

is_from_both_terriotries: bool = False

torch_fix_seed(seed)

atom_props_all: list[str] = list(atom_prop_dict_all.keys())
task_tokens: list[str] = tokenize(phi_task)
task_props: list[str] = [
    token for token in task_tokens if is_var(token, atom_prop_dict_all)
]
vars: tuple[str, ...] = tuple(
    [prop for prop in atom_props_all if prop not in task_props]
)

local_optima: list[SpecNode] = []
min_kl_divs: list[float] = []

for i in range(len(vars)):
    local_optimum, min_kl_div = search_single_start(
        search_depth,
        i,
        num_process,
        vars,
        phi_task,
        description,
        experiment,
        gpus,
        known_policy,
        num_sampling,
        num_replicates,
        total_timesteps,
        atom_prop_dict_all,
        obs_props_all,
        is_from_both_terriotries,
    )

    local_optima.append(local_optimum)
    min_kl_divs.append(min_kl_div)

local_optimum_specs: list[str] = nodes2specs(local_optima, phi_task, vars)
sorted_specs, sorted_min_kl_divs = sort_spec(local_optimum_specs, min_kl_divs)

multi_start_result = SortedLogger(experiment, sorted_specs, sorted_min_kl_divs)

multi_start_savename: str = "data/search/multistart_{}.json".format(experiment)

data_saver(multi_start_result, multi_start_savename)

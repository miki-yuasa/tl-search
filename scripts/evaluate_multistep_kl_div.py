from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
from numpy.typing import NDArray

from tl_search.tl.constants import (
    known_policy,
    unknown_policy,
    obs_props_all,
    atom_prop_dict_all,
)
from tl_search.tl.environment import create_env
from tl_search.tl.tl_generator import generate_tl_specs
from policies.simulator import (
    simulate_known_policy,
    create_comparison_domain,
    simulate_learned_policy,
)
from tl_search.common.typing import MultiStepKLDivLogger, PolicyMode
from tl_search.common.utils import (
    find_min_kl_div_tl_spec,
    kl_div,
)
from tl_search.common.io import (
    experiment_data_loader_all,
    list_model_names,
    data_saver,
)

policy_mode: PolicyMode = "known"
experiment: int = 5
total_processes: int = 16
num_replicates: int = 3
is_batched: bool = True

kl_div_steps: int = 1
num_sampling: int = 2000

# phi_task = "(psi_ba_ra & !psi_ba_rt)"
# specs = generate_tl_specs(phi_task, atom_prop_dict_all)

data = experiment_data_loader_all(policy_mode, experiment, is_batched, total_processes)

total_specs: int = len(data.tl_specs)
filenames: list[str] = list_model_names(
    policy_mode, experiment, total_processes, total_specs, num_replicates
)

tl_specs: list[str] = data.tl_specs

policy = known_policy if policy_mode == "known" else unknown_policy
map_locs_list, _ = policy.sample(num_sampling, False)

kl_divs: list[float] = []

for spec, filename in zip(tl_specs, filenames):
    env = create_env(spec, known_policy, obs_props_all, atom_prop_dict_all)
    env_monitored = Monitor(env)
    model = PPO.load(filename)

    similarities_spec: list[float] = []

    for map_locs in map_locs_list:
        actions_ori, action_probs_ori = simulate_known_policy(
            map_locs.blue_agent, map_locs.red_agent, known_policy, kl_div_steps
        )
        domain_ori: NDArray = create_comparison_domain(
            actions_ori, action_probs_ori, kl_div_steps
        )

        actions_tl, action_probs_tl = simulate_learned_policy(
            model, env, map_locs.blue_agent, map_locs.red_agent, kl_div_steps
        )
        domain_tl: NDArray = create_comparison_domain(
            actions_tl, action_probs_tl, kl_div_steps
        )

        similarity = kl_div(domain_ori, domain_tl)
        similarities_spec.append(similarity[0])

    kl_divs.append(np.mean(np.array(similarities_spec)))


(
    min_kl_div_spec,
    min_kl_div_mean,
    mean_kl_div_mean,
    max_kl_div_mean,
) = find_min_kl_div_tl_spec(
    data.tl_specs,
    kl_divs,
)

saved_data = MultiStepKLDivLogger(
    policy_mode,
    experiment,
    kl_div_steps,
    num_sampling,
    min_kl_div_spec,
    data.max_reward_spec,
    min_kl_div_mean,
    mean_kl_div_mean,
    max_kl_div_mean,
    data.min_reward_mean,
    data.mean_reward_mean,
    data.max_reward_mean,
    tl_specs,
    kl_divs,
    data.mean_rewards,
)

savename: str = "data/batch/multi-step_kl_div_{}_{}_{}_{}.json".format(
    policy_mode, experiment, kl_div_steps, num_sampling
)

data_saver(saved_data, savename)

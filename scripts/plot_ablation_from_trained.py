import pickle
from numpy.typing import NDArray

from tl_search.common.plotter import plot_ablation
from tl_search.envs.typing import EnemyPolicyMode

tuned_param_name: str = "ent_coef"
tuned_param_values: list[float] = [0.0, 0.01, 0.05, 0.1]
enemy_policy_modes: list[EnemyPolicyMode] = ["none"]  # "fight", "capture", "patrol"]

for enemy_policy_mode in enemy_policy_modes:
    print(f"enemy_policy_mode: {enemy_policy_mode}")
    for tuned_param_value in tuned_param_values:
        print(f"tuned_param_value: {tuned_param_value}")

        lc_data_filename: str = f"out/plots/reward_curve/heuristic/heuristic_enemy_{enemy_policy_mode}_ppo_{tuned_param_name}_{tuned_param_value}.pkl"

        save_path: str = f"out/plots/reward_curve/heuristic/heuristic_enemy_{enemy_policy_mode}_ppo_{tuned_param_name}_{tuned_param_value}.png"

        with open(lc_data_filename, "rb") as f:
            # X: list[NDArray] = pickle.load(f)
            # Y: list[NDArray] = pickle.load(f)
            lcs = pickle.load(f)

        plot_ablation(lcs, save_path, 5_000_000, 10000)

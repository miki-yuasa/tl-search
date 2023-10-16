from tl_search.envs.eval import evaluate_tuned_param_replicate_models
from tl_search.envs.typing import EnemyPolicyMode


enemy_policy_modes: list[EnemyPolicyMode] = ["none"]  # ["fight", "capture", "patrol"]
tuned_param_name: str = "ent_coef"
tuned_param_values: list[float] = [0.0, 0.01, 0.05]
num_episodes: int = 200
num_replicates: int = 3
map_path: str = "tl_search/map/maps/board_0002_obj.txt"
step_penalty_gamma: float = 0.01

for enemy_policy_mode in enemy_policy_modes:
    stats_path: str = (
        f"out/data/heuristic/stats/stats_{enemy_policy_mode}_enemy_ppo.json"
    )
    model_path: str = f"out/models/heuristic/{enemy_policy_mode}_enemy_ppo.zip"

    evaluate_tuned_param_replicate_models(
        tuned_param_name,
        tuned_param_values,
        num_replicates,
        enemy_policy_mode,
        model_path,
        map_path,
        num_episodes,
        stats_path,
        step_penalty_gamma=step_penalty_gamma,
    )

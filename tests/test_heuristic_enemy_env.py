import unittest, os

from stable_baselines3 import PPO
import seaborn as sns

from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.envs.train import simulate_model


class TestHeuristicEnemyEnv(unittest.TestCase):
    def test_heuristic_enemy_env(self):
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        file_path = "out/plots/field/test_heuristic_enemy_env.png"
        env = HeuristicEnemyEnv("fight", "tl_search/map/maps/board_0002_obj.txt")
        env.reset()
        env._is_red_agent_defeated = True
        env.render(file_path)

        self.assertTrue(os.path.exists(file_path))

    def test_heuristic_enemy_animation(self):
        model_path: str = (
            "out/models/heuristic/heuristic_enemy_fight_ppo_ent_coef_0.0_0.zip"
        )
        animation_path: str = "out/plots/animation/heuristic/test_heuristic_enemy.gif"
        env = HeuristicEnemyEnv("fight", "tl_search/map/maps/board_0002_obj.txt")
        model = PPO.load(model_path)
        simulate_model(model, env, animation_path)
        self.assertTrue(os.path.exists(animation_path))

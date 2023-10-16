from typing import Callable
import unittest, os

from stable_baselines3 import PPO
import seaborn as sns
from tl_search.common.typing import ObsProp

from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.map.utils import distance_area_point, distance_points


class TestTLMultigrid(unittest.TestCase):
    def test_render(self):
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        tl_spec: str = (
            "F(psi_ba_rf & !psi_ba_bt & !psi_ba_bf) & G(!psi_ba_ra & !psi_ra_bf)"
        )
        file_path = "out/plots/field/tl_multigrid.png"
        global_threshold: float = 0.5

        atom_prep_dict: dict[str, str] = {
            "psi_ba_ra": "d_ba_ra < {}".format(1.5),
            "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
            "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
            "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
            "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
        }

        obs_info: list[tuple[str, list[str], Callable]] = [
            (
                "d_ba_ra",
                ["blue_agent", "red_agent"],
                lambda x, y: distance_points(x, y),
            ),
            ("d_ba_rf", ["blue_agent", "red_flag"], lambda x, y: distance_points(x, y)),
            (
                "d_ba_bt",
                ["blue_agent", "blue_background"],
                lambda x, y: distance_area_point(x, y),
            ),
            (
                "d_ba_bf",
                ["blue_agent", "blue_flag"],
                lambda x, y: distance_points(x, y),
            ),
            ("d_ra_bf", ["red_agent", "blue_flag"], lambda x, y: distance_points(x, y)),
        ]
        obs_props: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]
        env = TLMultigrid(
            tl_spec,
            obs_props,
            atom_prep_dict,
            "fight",
            "tl_search/map/maps/board_0002_obj.txt",
        )
        env.reset()
        env._is_red_agent_defeated = False
        env.render(file_path)

        self.assertTrue(os.path.exists(file_path))

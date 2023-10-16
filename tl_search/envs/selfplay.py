from typing import Literal

from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

from tl_search.map.utils import mirror_flat_locs, mirror_loc
from tl_search.common.typing import Location
from tl_search.envs.multigrid import BaseMultigrid
from tl_search.envs.typing import ObsDict


class SelfPlay(BaseMultigrid):
    def __init__(
        self,
        enemy_policy_path: str,
        map_path: str,
        observation_space: spaces.Dict | None = None,
        is_move_clipped: bool = True,
        seed: int | None = None,
        randomness: float = 0.75,
        battle_reward_alpha: float = 0.25,
        obstacle_penalty_beta: float | None = None,
        step_penalty_gamma: float = 0,
        capture_reward: float = 1.0,
        num_max_steps: int = 500,
        render_mode: Literal["human", "rgb_array"] = "human",
    ) -> None:
        super().__init__(
            map_path,
            observation_space,
            is_move_clipped,
            seed,
            randomness,
            battle_reward_alpha,
            obstacle_penalty_beta,
            step_penalty_gamma,
            capture_reward,
            num_max_steps,
            render_mode,
        )

        self._enemy_policy_path = enemy_policy_path

    def _enemy_act(self) -> Location:
        enemy_model = PPO.load(self._enemy_policy_path)
        obs: ObsDict = self._reconstruct_enemy_obs(self._get_obs())
        action, _ = enemy_model.predict(obs)

        curr_loc: Location = Location(*obs["blue_agent"])
        next_loc: Location = self._act(curr_loc, int(action))

        out_loc: Location = Location(
            *mirror_loc(np.array(next_loc), self._field_map.shape)
        )

        return out_loc

    def _reconstruct_enemy_obs(self, obs: ObsDict) -> ObsDict:
        enemy_obs: ObsDict = {}

        enemy_obs["blue_agent"] = mirror_loc(obs["red_agent"], self._field_map.shape)
        enemy_obs["red_agent"] = mirror_loc(obs["blue_agent"], self._field_map.shape)
        enemy_obs["blue_flag"] = mirror_loc(obs["red_flag"], self._field_map.shape)
        enemy_obs["red_flag"] = mirror_loc(obs["blue_flag"], self._field_map.shape)
        enemy_obs["blue_background"] = mirror_flat_locs(
            obs["red_background"], self._field_map.shape
        )
        enemy_obs["red_background"] = mirror_flat_locs(
            obs["blue_background"], self._field_map.shape
        )
        enemy_obs["obstacle"] = mirror_flat_locs(obs["obstacle"], self._field_map.shape)
        enemy_obs["is_red_agent_defeated"] = obs["is_red_agent_defeated"]

        return enemy_obs

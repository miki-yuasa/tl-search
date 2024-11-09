from typing import Literal

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces
from safety_robot_gym.envs.fetch import MujocoBlockedFetchPickAndPlaceEnv
from safety_robot_gym.envs.fetch.pick_and_place import MODEL_XML_PATH

from tl_search.common.typing import ObsProp
from tl_search.tl.synthesis import TLAutomaton

obs_props: list[ObsProp] = [
    ObsProp("d_grip_tar", ["d_grip_tar"], lambda d_grip_tar: d_grip_tar),
    ObsProp("d_grip_ob", ["d_grip_ob"], lambda d_grip_ob: d_grip_ob),
    ObsProp("d_block_fall", ["d_block_fall"], lambda d_block_fall: d_block_fall),
]

default_distance_threshold: float = 0.05
atom_pred_dict: dict[str, str] = {
    "psi_grip_tar": f"d_grip_tar < {default_distance_threshold}",
    "psi_grip_ob": f"d_grip_ob < {default_distance_threshold}",
    "psi_block_fall": f"d_block_fall > {default_distance_threshold}",
}


class TLBlockedFetchPickAndPlaceEnv(MujocoBlockedFetchPickAndPlaceEnv):
    def __init__(
        self,
        tl_spec: str,
        obs_props: list[ObsProp] = obs_props,
        atom_pred_dict: dict[str, str] = atom_pred_dict,
        reward_type: Literal["sparse"] | Literal["dense"] = "dense",
        penalty_type: Literal["sparse"] | Literal["dense"] = "dense",
        dense_penalty_coef: float = 0.1,
        sparse_penalty_value: float = -100,
        max_episode_steps: int = 100,
        model_path: str = MODEL_XML_PATH,
        n_substeps: int = 20,
        gripper_extra_height: float = 0.2,
        target_in_the_air: bool = True,
        target_offset: float = 0,
        obj_range: float = 0.15,
        target_range: float = 0.15,
        distance_threshold: float = 0.05,
        render_mode: str | None = "rgb_array",
        **kwargs,
    ):
        self.aut = TLAutomaton(tl_spec, atom_pred_dict, obs_props)
        self._tl_spec: str = tl_spec
        self._obs_props: list[ObsProp] = obs_props
        self._atom_pred_dict: dict[str, str] = atom_pred_dict

        self._aut_state: int = self.aut.start

        super().__init__(
            reward_type,
            penalty_type,
            dense_penalty_coef,
            sparse_penalty_value,
            max_episode_steps,
            model_path,
            n_substeps,
            gripper_extra_height,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            **kwargs,
        )

    def _define_observation_space(self) -> spaces.Dict:
        obs = self._get_obs()
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64
                ),
                "aut_state": spaces.Discrete(self.aut.num_states),
            }
        )
        return observation_space

    def _get_obs(self) -> dict[str, NDArray[np.float64 | np.int_]]:
        obs: dict[str, NDArray[np.float64]] = super()._get_obs()
        obs["aut_state"] = np.array(self._aut_state)
        return obs

from typing import Literal, Any

import numpy as np
from numpy.typing import NDArray
from gymnasium import spaces
from safety_robot_gym.envs.fetch import MujocoBlockedFetchPushEnv
from safety_robot_gym.envs.fetch.push import MODEL_XML_PATH

from tl_search.common.typing import ObsProp, AutomatonStateStatus
from tl_search.tl.reward import tl_reward
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import tl2rob

obs_props: list[ObsProp] = [
    ObsProp("d_blk_tar", ["d_blk_tar"], lambda d_blk_tar: d_blk_tar),
    ObsProp("d_obs_moved", ["d_obs_moved"], lambda d_obs_moved: d_obs_moved),
    ObsProp("d_blk_fallen", ["d_blk_fallen"], lambda d_blk_fallen: d_blk_fallen),
]

default_distance_threshold: float = 0.05
atom_pred_dict: dict[str, str] = {
    "psi_blk_tar": f"d_blk_tar < {default_distance_threshold}",
    "psi_obs_moved": f"d_obs_moved < {default_distance_threshold}",
    "psi_blk_fallen": f"d_blk_fallen > {default_distance_threshold}",
}


class TLBlockedFetchPushEnv(MujocoBlockedFetchPushEnv):
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
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            **kwargs,
        )

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float64],
        desired_goal: NDArray[np.float64],
        info: dict,
    ) -> float:
        if isinstance(info, dict):
            info: list[dict[str, Any]] = [info]
        else:
            pass

        rewards: list[float] = []

        for info_dict in info:
            kin_dict: dict[str, NDArray[np.float64]] = {
                "d_blk_tar": info_dict["d_blk_tar"],
                "d_obs_moved": info_dict["d_obs_moved"],
                "d_blk_fallen": info_dict["d_blk_fallen"],
            }
            atom_rob_dict, obs_dict = atom_tl_ob2rob(self.aut, info_dict)
            reward, next_aut_state = tl_reward(
                atom_rob_dict, self.aut, info_dict["aut_state"], True
            )
            rewards.append(reward)

        # Change the shape of the rewards
        rewards_np: NDArray[np.float64] = np.array(rewards)
        rewards_np = rewards_np.reshape(-1, 1)

        output_reward = rewards_np[0] if rewards_np.size == 1 else rewards_np

        if rewards_np.size == 1:
            self._aut_state = next_aut_state
            self._aut_state_traj.append(next_aut_state)
            if next_aut_state in self.aut.goal_states:
                self._status = "goal"
            elif next_aut_state in self.aut.trap_states:
                self._status = "trap"
            else:
                self._status = "intermediate"
        else:
            pass

        return output_reward

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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: int | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._aut_state = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._aut_status: AutomatonStateStatus = "intermediate"
        return obs, info

    def _get_info(self) -> dict[str, Any]:
        info: dict[str, Any] = super()._get_info()
        info["aut_state"] = self._aut_state

        d_blk_tar: float = np.linalg.norm(info["goal"] - info["object_pos"])
        d_obs_moved: float = np.linalg.norm(
            info["obstacle_rel_pos"] - info["init_obstacle_pos"]
        )
        d_blk_fallen: float = -info["object_pos"][2]

        info["d_blk_tar"] = d_blk_tar
        info["d_obs_moved"] = d_obs_moved
        info["d_blk_fallen"] = d_blk_fallen

        return info

    def _is_success(self, achieved_goal: NDArray, desired_goal: NDArray) -> bool:
        return self._aut_state in self.aut.goal_states

    def compute_terminated(
        self,
        achieved_goal: NDArray[np.float64],
        desired_goal: NDArray[np.float64],
        info: dict[str, Any],
    ) -> bool:
        terminated: bool = super().compute_terminated(achieved_goal, desired_goal, info)

        in_trap_state: bool = self._aut_state in self.aut.trap_states

        return terminated or in_trap_state


def atom_tl_ob2rob(
    aut: TLAutomaton, kin_dict: dict[str, NDArray[np.float64] | np.float64]
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute robustnesses (rho) of the atomic TL porlocitions (psi) based on
    observation.

    Parameters
    ----------
    aut: TLAutomaton
        automaton from a TL spec
    kin_dict: dict[str, NDArray[np.float64]]
        Kinematic information of the agents

    Returns
    -------
    atom_rob_dict (dict[str, float]): dictionary of robustnesses of atomic propositions
    obs_dict (dict[str, float]): dictionary of observation and its values
    """

    obs_dict: dict[str, float] = {
        obs.name: obs.func(*[kin_dict[arg] for arg in obs.args])
        for obs in aut.obs_props
    }

    atom_rob_dict: dict[str, float] = {
        atom_props_key: tl2rob(aut.atom_prop_dict[atom_props_key], obs_dict)
        for atom_props_key in aut.atom_prop_dict.keys()
    }

    return (atom_rob_dict, obs_dict)

from typing import Callable

from highway_env.envs.common.action import Action
from highway_env.envs.parking_env import ParkingEnv
import numpy as np
from numpy.typing import NDArray

from tl_search.common.typing import AutomatonStateStatus, ObsProp
from tl_search.envs.parking import AdversarialParkingEnv
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import tl2rob


class TLAdversarialParkingEnv(AdversarialParkingEnv):
    obs_info: list[tuple[str, list[str], Callable]] = [("d_ego_goal")]

    def __init__(
        self,
        tl_spec: str,
        obs_props: list[ObsProp],
        atom_pred_dict: dict[str, str],
        config: dict | None = None,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__(config, render_mode)

        self._tl_spec: str = tl_spec
        self._obs_props: list[ObsProp] = obs_props
        self._atom_pred_dict: dict[str, str] = atom_pred_dict

        self.aut = TLAutomaton(tl_spec, atom_pred_dict, obs_props)

    def reset(self) -> tuple[dict[str, float], dict[str, float]]:
        obs, info = super().reset()

        self._aut_state: int = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._status: AutomatonStateStatus = "intermediate"

        return obs, info

    def compute_reward(
        self,
        achieved_goal: NDArray,
        desired_goal: NDArray,
        info: dict[str, float | NDArray],
        p: float = 0.5,
    ) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        # obs = obs if isinstance(obs, tuple) else (obs,)
        reward = self.compute_reward(
            obs["achieved_goal"],
            obs["desired_goal"],
            self._info(self.observation_type.observe(), action),
        )
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )

        # if self.goal.hit:
        #     reward += 0.12

        return reward

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)

        achieved_goal = obs["achieved_goal"]

        obs = self.observation_type_parking.observe()
        success = self._is_success(achieved_goal, obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def _is_success(self, achieved_goal: NDArray, desired_goal: NDArray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )


def atom_tl_ob2rob(
    aut: TLAutomaton, kin_dict: dict[str, NDArray[np.float_]]
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute robustnesses (rho) of the atomic TL porlocitions (psi) based on
    observation.

    Parameters
    ----------
    aut: TLAutomaton
        automaton from a TL spec
    kin_dict: dict[str, NDArray[np.float_]]
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

from typing import Any, Callable

from gymnasium.spaces.space import Space
from gymnasium import spaces
from highway_env.envs.common.action import action_factory
from highway_env.envs.parking_env import ParkingEnv
import numpy as np
from numpy.typing import NDArray

from tl_search.common.typing import AutomatonStateStatus, ObsProp
from tl_search.envs.parking import (
    AdversarialParkingEnv,
    KinematicGoalVehiclesObservation,
)
from tl_search.tl.reward import tl_reward
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import tl2rob


class TLAdversarialParkingEnv(AdversarialParkingEnv):
    obs_info: list[tuple[str, list[str], Callable]] = [
        ("d_ego_goal", ["d_ego_goal"], lambda d_ego_goal: d_ego_goal),
        ("d_ego_adv", ["d_ego_adv"], lambda d_ego_adv: d_ego_adv),
        ("d_ego_wall", ["d_ego_wall"], lambda d_ego_wall: d_ego_wall),
    ]
    obs_props: list[ObsProp] = [
        ObsProp(name, args, func) for name, args, func in obs_info
    ]

    atom_pred_dict: dict[str, str] = {
        "psi_ego_goal": "d_ego_goal < {}".format(1),
        "psi_ego_adv": "d_ego_adv < {}".format(3),
        "psi_ego_wall": "d_ego_wall < {}".format(3),
    }

    def __init__(
        self,
        tl_spec: str,
        config: dict | None = None,
        obs_props: list[ObsProp] | None = None,
        atom_pred_dict: dict[str, str] | None = None,
        render_mode: str | None = "rgb_array",
    ) -> None:

        self.aut = TLAutomaton(tl_spec, self.atom_pred_dict, self.obs_props)

        super().__init__(config, render_mode)

        self._tl_spec: str = tl_spec
        self._obs_props = obs_props if obs_props is not None else self.obs_props
        self._atom_pred_dict = (
            atom_pred_dict if atom_pred_dict is not None else self.atom_pred_dict
        )
        self._dense_reward: bool = self.config["dense_reward"]

    def define_spaces(self) -> None:
        self.observation_type = TLKinematicGoalVehiclesObservation(
            self, **self.config["observation"]
        )
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()
        self.observation_type_parking = TLKinematicGoalVehiclesObservation(
            self, **self.config["observation"]
        )

    def reset(self, *args, **kwargs) -> tuple[dict[str, float], dict[str, float]]:
        self._aut_state: int = self.aut.start
        obs, info = super().reset(*args, **kwargs)

        self._aut_state: int = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._status: AutomatonStateStatus = "intermediate"

        return obs, info

    def compute_reward(
        self,
        achieved_goal: NDArray[np.float_],
        desired_goal: NDArray[np.float_],
        info: dict[str, Any] | list[dict[str, Any]],
        p: float = 0.5,
    ) -> float | NDArray[np.float_]:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """

        # Create kinematic information dictionary
        infos = info if hasattr(info, "size") else [info]

        num_features: int = len(self.config["observation"]["features"])

        adversarial_agent_locs_tmp: list[NDArray[np.float_]] = [
            info["adversarial_agent_obs"][0:2] for info in infos
        ]
        aut_states: list[int] = [info["aut_state"] for info in infos]

        adv_locs: NDArray[np.float_] = np.array(adversarial_agent_locs_tmp)
        ego_locs: NDArray[np.float_] = achieved_goal.reshape(-1, num_features)[:, 0:2]
        goal_locs: NDArray[np.float_] = desired_goal.reshape(-1, num_features)[:, 0:2]

        # Compute the distance between the goal and the achieved goal
        d_ego_goal: NDArray[np.float_] = np.linalg.norm(ego_locs - goal_locs, axis=-1)

        # Compute the distance between the ego and the adversarial agents
        d_ego_adv: NDArray[np.float_] = np.linalg.norm(ego_locs - adv_locs, axis=-1)

        # Compute the distance between the ego and walls
        wall_width, wall_height = self.wall_width, self.wall_height
        wall_min_x, wall_max_x = -wall_width / 2, wall_width / 2
        wall_min_y, wall_max_y = -wall_height / 2, wall_height / 2

        d_ego_wall = np.array(
            np.min(
                np.concatenate(
                    [
                        np.abs(ego_locs[:, 0] - wall_min_x).reshape([-1, 1]),
                        np.abs(ego_locs[:, 0] - wall_max_x).reshape([-1, 1]),
                        np.abs(ego_locs[:, 1] - wall_min_y).reshape([-1, 1]),
                        np.abs(ego_locs[:, 1] - wall_max_y).reshape([-1, 1]),
                    ],
                    axis=-1,
                ),
                axis=-1,
            )
        ).reshape(*d_ego_adv.shape)

        rewards: list[float] = []

        for i in range(len(infos)):
            kin_dict = {
                "d_ego_goal": d_ego_goal[i],
                "d_ego_adv": d_ego_adv[i],
                "d_ego_wall": d_ego_wall[i],
            }

            atom_rob_dict, obs_dict = atom_tl_ob2rob(self.aut, kin_dict)
            reward, next_aut_state = tl_reward(
                atom_rob_dict, self.aut, aut_states[i], self._dense_reward
            )

            rewards.append(reward)

        # Change the shape of the rewards
        rewards_np: NDArray[np.float_] = np.array(rewards)
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

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        reward = self.compute_reward(
            obs["achieved_goal"],
            obs["desired_goal"],
            self._info(self.observation_type.observe(), action),
        )

        return reward

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)

        achieved_goal = obs["achieved_goal"]

        obs = self.observation_type_parking.observe()
        success = self._is_success(achieved_goal, obs["desired_goal"])
        info.update({"is_success": success})
        info.update(
            {
                "adversarial_agent_obs": obs["observation"].reshape(
                    -1, len(self.config["observation"]["features"])
                )[1, :]
            }
        )
        info.update({"aut_state": self._aut_state})
        return info

    def _is_success(self, achieved_goal: NDArray, desired_goal: NDArray) -> bool:
        return self._aut_state in self.aut.goal_states

    def _is_terminated(self) -> bool:
        terminated: bool = super()._is_terminated()

        in_trap_state: bool = self._aut_state in self.aut.trap_states

        return terminated or in_trap_state


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


class TLKinematicGoalVehiclesObservation(KinematicGoalVehiclesObservation):
    def space(self) -> Space:
        obs = super().observe()
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["observation"].shape,
                    dtype=np.float64,
                ),
                "achieved_goal": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["achieved_goal"].shape,
                    dtype=np.float64,
                ),
                "desired_goal": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["desired_goal"].shape,
                    dtype=np.float64,
                ),
                "aut_state": spaces.Discrete(self.env.aut.num_states),
            }
        )

    def observe(self) -> dict[str, Any]:
        obs = super().observe()
        obs["aut_state"] = self.env._aut_state
        return obs

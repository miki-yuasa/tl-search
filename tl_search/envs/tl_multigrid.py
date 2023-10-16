from typing import Literal, TypeAlias, TypedDict
from gymnasium import spaces
import gymnasium as gym
import numpy as np

from tl_search.common.typing import AutomatonStateStatus, Location, ObsProp
from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.envs.multigrid import BaseMultigrid
from tl_search.envs.typing import EnemyPolicyMode, FieldObj, InfoDict, ObsDict
from tl_search.map.utils import distance_points
from tl_search.policies.probabilistic import CapturePolicy, FightPolicy, PatrolPolicy
from tl_search.tl.reward import tl_reward
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import atom_tl_ob2rob, tl2rob

EnemyPolicy: TypeAlias = FightPolicy | PatrolPolicy | CapturePolicy


class TLMultigridTmp(BaseMultigrid):
    # The Element IDs
    # BLUE_BACKGROUND = 0
    # RED_BACKGROUND = 1
    # BLUE_UGV = 2
    # BLUE_UAV = 3
    # RED_UGV = 4
    # RED_UAV = 5
    # BLUE_FLAG = 6
    # RED_FLAG = 7
    # OBSTACLE = 8
    def __init__(
        self,
        tl_spec: str,
        obs_props: list[ObsProp],
        atom_prep_dict: dict[str, str],
        enemy_policy_mode: EnemyPolicyMode,
        map_path: str,
        observation_space: spaces.Dict | None = None,
        is_move_clipped: bool = True,
        num_max_steps: int = 200,
        render_mode: Literal["human", "rgb_array"] = "human",
    ) -> None:
        super().__init__(
            map_path,
            observation_space,
            is_move_clipped,
            num_max_steps=num_max_steps,
            render_mode=render_mode,
        )

        self._tl_spec: str = tl_spec
        self._obs_props: list[ObsProp] = obs_props
        self._atom_prep_dict: dict[str, str] = atom_prep_dict
        self._enemy_policy_mode: EnemyPolicyMode = enemy_policy_mode

        self.aut = TLAutomaton(tl_spec, atom_prep_dict, obs_props)

        self._enemy_policy: EnemyPolicy

        match enemy_policy_mode:
            case "capture":
                self._enemy_policy = CapturePolicy(
                    self._fixed_obj, self._field_map, self._randomness
                )
            case "fight":
                self._enemy_policy = FightPolicy(
                    self._fixed_obj, self._field_map, self._randomness
                )
            case "patrol":
                self._enemy_policy = PatrolPolicy(
                    self._fixed_obj, self._field_map, self._randomness
                )
            case "none":
                pass
            case _:
                raise Exception(
                    f"[tl_search] The enemy policy {enemy_policy_mode} is not defined."
                )

    def reset(
        self,
        blue_agent_loc: Location | None = None,
        red_agent_loc: Location | None = None,
        is_red_agent_defeated: bool = False,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[ObsDict, InfoDict]:
        super().reset(
            blue_agent_loc,
            red_agent_loc,
            is_red_agent_defeated,
            seed=seed,
            options=options,
        )
        self._enemy_policy.reset(self._red_agent_loc)
        self._aut_state = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._last_successful_aut_state = self.aut.start
        self._status: AutomatonStateStatus = "intermediate"

        obs: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list = [obs]

        return obs, info

    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]:
        observation, reward, terminated, truncated, info = super().step(action)

        self._field = FieldObj(
            self._blue_agent_loc, self._red_agent_loc, *self._fixed_obj
        )

        return observation, reward, terminated, truncated, info

    def _enemy_act(self) -> Location:
        loc: Location

        match self._enemy_policy:
            case CapturePolicy():
                loc = self._enemy_policy.act()
            case FightPolicy():
                loc = self._enemy_policy.act(self._blue_agent_loc)
            case PatrolPolicy():
                loc = self._enemy_policy.act()
            case _:
                raise Exception(
                    f"[tl_search] The enemy policy {self._enemy_policy_mode} is not defined."
                )
        return loc

    def _reward(self) -> tuple[float, bool, bool]:
        atom_rob_dict, obs_dict = atom_tl_ob2rob(
            self.aut,
            FieldObj(
                self._blue_agent_loc,
                self._red_agent_loc,
                *self._fixed_obj,
                self._is_red_agent_defeated,
            ),
        )
        reward, self._aut_state = tl_reward(atom_rob_dict, self.aut, self._aut_state)

        self._last_successful_aut_state = (
            self._aut_state
            if self._aut_state not in self.aut.trap_states
            else self._last_successful_aut_state
        )

        terminated: bool = False
        truncated: bool = False

        status: AutomatonStateStatus
        if self._aut_state in [*self.aut.goal_states, *self.aut.trap_states]:
            terminated = True
            status = "goal" if self._aut_state in self.aut.goal_states else "trap"

        elif self._aut_state in list(
            range(self.aut.edges[-1].state + 1)
        ):  # or self.aut_state == 1:
            terminated = False
            status = "intermediate"
            if not self._is_red_agent_defeated:
                self._is_red_agent_defeated = (
                    True
                    if tl2rob("psi_ba_ra & psi_ba_bt", atom_rob_dict) >= 0
                    else False
                )
        else:
            raise Exception(
                f"[tl_search] The automaton state {self._aut_state} is not defined."
            )

        if not terminated and (
            self._red_agent_loc == self._fixed_obj.blue_flag
            or self._blue_agent_loc == self._fixed_obj.red_flag
        ):
            truncated = True
        else:
            pass

        self._status = status

        return reward, terminated, truncated


class TLMultigrid(HeuristicEnemyEnv):
    def __init__(
        self,
        tl_spec: str,
        obs_props: list[ObsProp],
        atom_prep_dict: dict[str, str],
        enemy_policy_name: EnemyPolicyMode,
        map_path: str,
        observation_space: spaces.Dict | None = None,
        is_move_clipped: bool = True,
        num_max_steps: int = 200,
        render_mode: Literal["human", "rgb_array"] = "human",
        randomness: float = 0.75,
        battle_reward_alpha: float = 0.25,
        obstacle_penalty_beta: float | None = None,
        step_penalty_gamma: float = 0,
    ) -> None:
        super().__init__(
            enemy_policy_name,
            map_path,
            observation_space,
            is_move_clipped,
            randomness,
            battle_reward_alpha,
            obstacle_penalty_beta,
            step_penalty_gamma,
            num_max_steps,
            render_mode,
        )

        self._tl_spec: str = tl_spec
        self._obs_props: list[ObsProp] = obs_props
        self._atom_prep_dict: dict[str, str] = atom_prep_dict

        self.aut = TLAutomaton(tl_spec, atom_prep_dict, obs_props)

    def reset(
        self,
        blue_agent_loc: Location | None = None,
        red_agent_loc: Location | None = None,
        is_red_agent_defeated: bool = False,
        seed: int | None = None,
    ) -> tuple[ObsDict, InfoDict]:
        obs, info = super().reset(blue_agent_loc, red_agent_loc, seed)

        self._is_red_agent_defeated: bool = is_red_agent_defeated

        self._aut_state = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._last_successful_aut_state = self.aut.start
        self._status: AutomatonStateStatus = "intermediate"
        self._field = FieldObj(
            self._blue_agent_loc, self._red_agent_loc, *self._fixed_obj
        )

        return obs, info

    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]:
        observation, reward, terminated, truncated, info = super().step(action)

        self._field = FieldObj(
            self._blue_agent_loc, self._red_agent_loc, *self._fixed_obj
        )

        return observation, reward, terminated, truncated, info

    def _reward(self) -> tuple[float, bool, bool]:
        atom_rob_dict, obs_dict = atom_tl_ob2rob(
            self.aut,
            FieldObj(
                self._blue_agent_loc,
                self._red_agent_loc,
                *self._fixed_obj,
                self._is_red_agent_defeated,
            ),
        )
        reward, self._aut_state = tl_reward(atom_rob_dict, self.aut, self._aut_state)

        self._last_successful_aut_state = (
            self._aut_state
            if self._aut_state not in self.aut.trap_states
            else self._last_successful_aut_state
        )

        terminated: bool = False
        truncated: bool = self._step_count >= self._num_max_steps

        if self._blue_agent_loc == self._fixed_obj.red_flag:
            terminated = True
        else:
            pass

        if self._red_agent_loc == self._fixed_obj.blue_flag:
            terminated = True
        else:
            pass

        if (
            distance_points(self._blue_agent_loc, self._red_agent_loc) <= 1
            and not self._is_red_agent_defeated
        ):
            blue_win: bool

            ba_in_bt: bool = self._blue_agent_loc in self._fixed_obj.blue_background
            ra_in_rt: bool = self._red_agent_loc in self._fixed_obj.red_background

            # match (ba_in_bt, ra_in_rt):
            #     case (True, False):
            #         blue_win = np.random.choice(
            #             [True, False], p=[self._randomness, 1.0 - self._randomness]
            #         )
            #     case (False, True):
            #         blue_win = np.random.choice(
            #             [False, True], p=[self._randomness, 1.0 - self._randomness]
            #         )

            #     case (True, True):
            #         blue_win = np.random.choice([True, False], p=[0.5, 0.5])

            #     case (False, False):
            #         raise Exception("[tl_search] Both agents are out of the game.")

            match self._blue_agent_loc in self._fixed_obj.blue_background:
                case True:
                    blue_win = np.random.choice(
                        [True, False], p=[self._randomness, 1.0 - self._randomness]
                    )
                case False:
                    blue_win = np.random.choice(
                        [False, True], p=[self._randomness, 1.0 - self._randomness]
                    )

            if blue_win:
                self._is_red_agent_defeated = True
            else:
                terminated = True

        if self._obstacle_penalty_beta is not None:
            if self._blue_agent_loc in self._fixed_obj.obstacle:
                terminated = True
            else:
                pass
        else:
            pass

        if self._aut_state in [*self.aut.goal_states, *self.aut.trap_states]:
            terminated = True
            self._status = "goal" if self._aut_state in self.aut.goal_states else "trap"

        elif self._aut_state in list(
            range(self.aut.edges[-1].state + 1)
        ):  # or self.aut_state == 1:
            terminated = False
            self._status = "intermediate"
        else:
            raise Exception(
                f"[tl_search] The automaton state {self._aut_state} is not defined."
            )

        return reward, terminated, truncated


def create_env(
    tl_spec: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    observation_space: spaces.Dict | None = None,
    is_move_clipped: bool = True,
    num_max_steps: int = 500,
    render_mode: Literal["human", "rgb_array"] = "human",
) -> TLMultigrid:
    env = TLMultigrid(
        tl_spec,
        obs_props,
        atom_prep_dict,
        enemy_policy_mode,
        map_path,
        observation_space,
        is_move_clipped,
        num_max_steps,
        render_mode,
    )

    return env


def make_env(
    tl_spec: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    observation_space: spaces.Dict | None = None,
    is_move_clipped: bool = True,
    num_max_steps: int = 500,
    render_mode: Literal["human", "rgb_array"] = "human",
):
    def _init() -> gym.Env:
        return create_env(
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            observation_space,
            is_move_clipped,
            num_max_steps,
            render_mode,
        )

    return _init


class TLMultigridDefaultArgs(TypedDict):
    observation_space: spaces.Dict | None
    is_move_clipped: bool
    num_max_steps: int
    render_mode: Literal["human", "rgb_array"]

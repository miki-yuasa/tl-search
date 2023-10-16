import sys, random
from typing import Union, cast

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import gym
from gym import spaces

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
import torch

from tl_search.tl.tl_parser import atom_tl_ob2rob, get_used_obs, get_vars, tokenize
from tl_search.map.utils import parse_map
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.reward import tl_reward
from tl_search.policies.known_policy import KnownPolicy
from tl_search.policies.heuristic import RandomPolicy, AStarPolicy
from tl_search.policies.unknown_policy import UnknownPolicy
from tl_search.common.typing import (
    AutomatonStateStatus,
    EnemyPolicyMode,
    Location,
    MapLocations,
    MapObjectIDs,
    FixedMapLocations,
    Action,
    EnvProps,
    MapProps,
    ModelProps,
    ObsProp,
)


# class Environment(gym.Env):
#     """Environment for the CtF game."""

#     # The Element IDs
#     # BLUE_BACKGROUND = 0
#     # RED_BACKGROUND = 1
#     # BLUE_UGV = 2
#     # BLUE_UAV = 3
#     # RED_UGV = 4
#     # RED_UAV = 5
#     # BLUE_FLAG = 6
#     # RED_FLAG = 7
#     # OBSTACLE = 8

#     def __init__(
#         self,
#         n_in: int,
#         init_map: NDArray,
#         map_object_ids: MapObjectIDs,
#         aut: TLAutomaton,
#         enemy_policy: EnemyPolicyMode,
#         is_initialized_both_territories: bool,
#     ) -> None:
#         # Memoize the locations of  elements
#         # The locations are sotred like [(0,2), (2,1), ...]

#         self.map_object_ids: MapObjectIDs = map_object_ids

#         self.fixed_map_locs: FixedMapLocations = cast(
#             FixedMapLocations, parse_map(init_map, self.map_object_ids)
#         )

#         self.is_initialized_both_territories: bool = is_initialized_both_territories

#         (
#             self.blue_flag,
#             self.red_flag,
#             self.blue_territory,
#             self.red_territory,
#             self.obstacles,
#             self.walls,
#         ) = self.fixed_map_locs

#         self.blue_agent: Location = (
#             random.choice(self.blue_territory + self.red_territory)
#             if self.is_initialized_both_territories
#             else random.choice(self.blue_territory)
#         )
#         self.red_agent: Location = (
#             random.choice(self.blue_territory + self.red_territory)
#             if self.is_initialized_both_territories
#             else random.choice(self.red_territory)
#         )

#         # Initialize the automaton state.
#         self.aut: TLAutomaton = aut
#         self.aut_state: int = self.aut.start
#         self.last_successful_aut_state: int = self.aut.start

#         # Store the initial info
#         self.init_info: dict = self.save_info()

#         self.enemy_policy = enemy_policy
#         self.red_policy = (
#             AStarPolicy(
#                 self.red_agent,
#                 self.blue_flag,
#                 init_map,
#             )
#             if enemy_policy == "a-star"
#             else RandomPolicy(self.red_agent, self.fixed_map_locs)
#         )
#         self.red_path = self.red_policy.path

#         self.blue_path = [self.blue_agent]
#         self.init_map = init_map
#         self.action_probs: list[list[float]] = []
#         self.actions: list[int] = []

#         self.observation_space: spaces.Box = spaces.Box(
#             low=-10, high=10, shape=(n_in,), dtype=np.float32
#         )
#         self.action_space: spaces.Discrete = spaces.Discrete(5)

#     def reset(self) -> NDArray:
#         blue_agent: Location = (
#             random.choice(self.blue_territory + self.red_territory)
#             if self.is_initialized_both_territories
#             else random.choice(self.blue_territory)
#         )
#         red_agent: Location = (
#             random.choice(self.blue_territory + self.red_territory)
#             if self.is_initialized_both_territories
#             else random.choice(self.red_territory)
#         )

#         obs = self.reset_with_agent_locs(blue_agent, red_agent)
#         return obs

#     def reset_with_agent_locs(
#         self, blue_agent: Location, red_agent: Location
#     ) -> NDArray:
#         """
#         Reset the environment to the initial state and return the initial observation.

#         Parameters:

#         None

#         Returms

#         init_observation (NDArray[d_BA_RF, d_BA_RR, d_BA_RA, d_RA_BF, d_BA_Ob]):
#                 observation of the initial state
#         """

#         _, locations = self.init_info.values()

#         (
#             _,
#             _,
#             self.blue_flag,
#             self.red_flag,
#             self.blue_territory,
#             self.red_territory,
#             self.obstacles,
#             self.walls,
#         ) = locations

#         self.blue_agent = blue_agent
#         self.red_agent = red_agent
#         self.red_policy = (
#             AStarPolicy(
#                 self.red_agent,
#                 self.blue_flag,
#                 self.init_map,
#             )
#             if self.enemy_policy == "a-star"
#             else RandomPolicy(self.red_agent, self.fixed_map_locs)
#         )
#         self.red_path = self.red_policy.path
#         self.blue_path = [self.blue_agent]

#         self.last_successful_aut_state = self.aut.start
#         self.aut_state = self.aut.start
#         self.action_probs = []
#         self.actions = []

#         info = self.save_info()
#         _, obs_dict = atom_tl_ob2rob(self.aut, info["locations"])

#         return np.array(list(obs_dict.values())).astype(np.float32)

#     def step(self, action: Action) -> tuple[object, float, bool, dict]:
#         """
#         Updates the environment by the given action and returns a tuple of next
#         state info

#         Parameters:

#          action (int): an action provided by the agent

#         Returns:
#         -------
#         observation (object): agent's observation of the current environment
#         reward (float) : amount of reward returned after previous action
#         done (bool): whether the episode has ended, in which case further step()
#                      calls will return undefined results
#         info (dict): contains auxiliary diagnostic information (helpful for debugging,
#                      and sometimes for learning)
#         """

#         self.blue_agent = act(self.blue_agent, action)
#         self.red_agent = self.red_policy.act()

#         info = self.save_info()
#         atom_rob_dict, obs_dict = atom_tl_ob2rob(self.aut, info["locations"])
#         obs_next = np.array(list(obs_dict.values())).astype(np.float32)
#         rhos_next = list(atom_rob_dict.values())

#         reward: float
#         reward, self.aut_state = tl_reward(atom_rob_dict, self.aut, self.aut_state)

#         self.last_successful_aut_state = (
#             self.aut_state
#             if self.aut_state not in self.aut.trap_states
#             else self.last_successful_aut_state
#         )

#         # if not self.aut_state == self.aut.start:  # 0:
#         #    self.red_agent = Location(-1, -1)
#         # else:
#         #    pass

#         self.blue_path.append(self.blue_agent)
#         self.red_path.append(self.red_agent)

#         done: bool  # Initialize the done bool
#         status: AutomatonStateStatus
#         # if self.aut_state == 1 or self.aut_state == -1:
#         if self.aut_state in [*self.aut.goal_states, *self.aut.trap_states]:
#             done = True
#             status = "goal" if self.aut_state in self.aut.goal_states else "trap"
#         elif self.aut_state in list(
#             range(self.aut.edges[-1].state + 1)
#         ):  # or self.aut_state == 1:
#             done = False
#             status = "intermediate"
#         else:
#             print(
#                 "Error: Unknow automaton state. Cannot determine if it is done or not.",
#                 file=sys.stderr,
#             )
#             sys.exit(1)

#         if self.red_agent == self.blue_flag or self.blue_agent == self.red_flag:
#             done = True
#         else:
#             pass

#         # Construct diagonostic info dict
#         locations: MapLocations = MapLocations(
#             self.blue_agent,
#             self.red_agent,
#             self.blue_flag,
#             self.red_flag,
#             self.blue_territory,
#             self.red_territory,
#             self.obstacles,
#             self.walls,
#         )
#         all_info: dict = {
#             "rhos": rhos_next,
#             "aut_state": self.aut_state,
#             "last_successful_aut_state": self.last_successful_aut_state,
#             "locations": locations,
#             "aut_state_status": status,
#         }

#         return (obs_next, reward, done, all_info)

#     def save_info(self) -> dict:
#         locations: MapLocations = MapLocations(
#             self.blue_agent,
#             self.red_agent,
#             self.blue_flag,
#             self.red_flag,
#             self.blue_territory,
#             self.red_territory,
#             self.obstacles,
#             self.walls,
#         )
#         info: dict = {
#             "aut_state": self.aut_state,
#             "locations": locations,
#         }

#         return info

#     def save_action_prob(self, action_prob: list[float]):
#         self.action_probs.append(action_prob)

#     def save_action(self, action: int):
#         self.actions.append(action)

#     def render(self, mode: str = "console"):
#         if mode != "console":
#             raise NotImplementedError()

#         _, ax = plt.subplots()
#         h, w = self.init_map.shape

#         markersize = 30
#         ax.plot(
#             self.blue_flag.x,
#             self.blue_flag.y,
#             marker=">",
#             color="b",
#             markersize=markersize,
#         )
#         ax.plot(
#             self.red_flag.x,
#             self.red_flag.y,
#             marker=">",
#             color="firebrick",
#             markersize=markersize,
#         )
#         ax.set_xlim(0, w - 0.5)
#         ax.set_ylim(0, h - 0.5)
#         ax.set_aspect(1)
#         ax.invert_yaxis()

#         for obs in self.obstacles:
#             obs_rec = plt.Rectangle((obs.x, obs.y), 1, 1, color="black")
#             ax.add_patch(obs_rec)

#         for bt in self.blue_territory:
#             bt_rec = plt.Rectangle((bt.x, bt.y), 1, 1, color="cornflowerblue")
#             ax.add_patch(bt_rec)

#         for rt in self.red_territory:
#             rt_rec = plt.Rectangle((rt.x, rt.y), 1, 1, color="lightcoral")
#             ax.add_patch(rt_rec)

#         blue_agent_fight_region = plt.Circle(
#             (self.blue_agent.x, self.blue_agent.y),
#             2,
#             fill=False,
#             ec="grey",
#             label="Fight region",
#         )
#         ax.add_artist(blue_agent_fight_region)
#         ax.plot(
#             self.blue_agent.x,
#             (self.blue_agent.y),
#             marker="o",
#             color="b",
#             markersize=markersize,
#             label="Blue agent",
#         )
#         ax.plot(
#             self.red_agent.x,
#             (self.red_agent.y),
#             marker="o",
#             color="firebrick",
#             markersize=markersize,
#         )
#         plt.show(block=False)


class Environment(gym.Env):
    """Environment for the CtF game."""

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
        env_props: EnvProps,
        is_initialized_both_territories: bool,
    ) -> None:
        # Memoize the locations of  elements
        # The locations are sotred like [(0,2), (2,1), ...]

        self.map_object_ids: MapObjectIDs = env_props.map_props.map_object_ids

        self.fixed_map_locs: FixedMapLocations = env_props.map_props.fixed_map_locs

        self.init_map: NDArray = env_props.map_props.map

        self.is_initialized_both_territories: bool = is_initialized_both_territories

        (
            self.blue_flag,
            self.red_flag,
            self.blue_territory,
            self.red_territory,
            self.obstacles,
            self.walls,
        ) = self.fixed_map_locs

        self.blue_agent: Location = (
            random.choice(self.blue_territory + self.red_territory)
            if self.is_initialized_both_territories
            else random.choice(self.blue_territory)
        )
        self.red_agent: Location = (
            random.choice(self.blue_territory + self.red_territory)
            if self.is_initialized_both_territories
            else random.choice(self.red_territory)
        )

        # Initialize the automaton state.
        self.aut = TLAutomaton(
            env_props.spec, env_props.atom_prop_dict, env_props.obs_props
        )
        self.aut_state: int = self.aut.start
        self.last_successful_aut_state: int = self.aut.start

        # Store the initial info
        self.init_info: dict = self.save_info()

        self.enemy_policy = env_props.enemy_policy
        self.red_policy = (
            AStarPolicy(
                self.red_agent,
                self.blue_flag,
                self.init_map,
            )
            if self.enemy_policy == "a-star"
            else RandomPolicy(self.red_agent, self.fixed_map_locs)
        )
        self.red_path = self.red_policy.traj

        self.blue_path = [self.blue_agent]
        self.action_probs: list[list[float]] = []
        self.actions: list[int] = []

        self.observation_space: spaces.Box = spaces.Box(
            low=-10, high=10, shape=(len(env_props.obs_props),), dtype=np.float32
        )
        self.action_space: spaces.Discrete = spaces.Discrete(5)
        self.render_mode: str = "human"

    def reset(self) -> NDArray:
        blue_agent: Location = (
            random.choice(self.blue_territory + self.red_territory)
            if self.is_initialized_both_territories
            else random.choice(self.blue_territory)
        )
        red_agent: Location = (
            random.choice(self.blue_territory + self.red_territory)
            if self.is_initialized_both_territories
            else random.choice(self.red_territory)
        )

        obs = self.reset_with_agent_locs(blue_agent, red_agent)
        return obs

    def reset_with_agent_locs(
        self, blue_agent: Location, red_agent: Location
    ) -> NDArray:
        """
        Reset the environment to the initial state and return the initial observation.

        Parameters:

        None

        Returms

        init_observation (NDArray[d_BA_RF, d_BA_RR, d_BA_RA, d_RA_BF, d_BA_Ob]):
                observation of the initial state
        """

        _, locations = self.init_info.values()

        (
            _,
            _,
            self.blue_flag,
            self.red_flag,
            self.blue_territory,
            self.red_territory,
            self.obstacles,
            self.walls,
        ) = locations

        self.blue_agent = blue_agent
        self.red_agent = red_agent
        self.red_policy = (
            AStarPolicy(
                self.red_agent,
                self.blue_flag,
                self.init_map,
            )
            if self.enemy_policy == "a-star"
            else RandomPolicy(self.red_agent, self.fixed_map_locs)
        )
        self.red_path = self.red_policy.traj
        self.blue_path = [self.blue_agent]

        self.last_successful_aut_state = self.aut.start
        self.aut_state = self.aut.start
        self.action_probs = []
        self.actions = []

        info = self.save_info()
        _, obs_dict = atom_tl_ob2rob(self.aut, info["locations"])

        return np.array(list(obs_dict.values())).astype(np.float32)

    def step(self, action: Action) -> tuple[object, float, bool, dict]:
        """
        Updates the environment by the given action and returns a tuple of next
        state info

        Parameters:

         action (int): an action provided by the agent

        Returns:
        -------
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step()
                     calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging,
                     and sometimes for learning)
        """

        self.blue_agent = act(self.blue_agent, action)
        self.red_agent = self.red_policy.act()

        info = self.save_info()
        atom_rob_dict, obs_dict = atom_tl_ob2rob(self.aut, info["locations"])
        obs_next = np.array(list(obs_dict.values())).astype(np.float32)
        rhos_next = list(atom_rob_dict.values())

        reward: float
        reward, self.aut_state = tl_reward(atom_rob_dict, self.aut, self.aut_state)

        self.last_successful_aut_state = (
            self.aut_state
            if self.aut_state not in self.aut.trap_states
            else self.last_successful_aut_state
        )

        # if not self.aut_state == self.aut.start:  # 0:
        #    self.red_agent = Location(-1, -1)
        # else:
        #    pass

        self.blue_path.append(self.blue_agent)
        self.red_path.append(self.red_agent)

        done: bool  # Initialize the done bool
        status: AutomatonStateStatus
        # if self.aut_state == 1 or self.aut_state == -1:
        if self.aut_state in [*self.aut.goal_states, *self.aut.trap_states]:
            done = True
            status = "goal" if self.aut_state in self.aut.goal_states else "trap"
        elif self.aut_state in list(
            range(self.aut.edges[-1].state + 1)
        ):  # or self.aut_state == 1:
            done = False
            status = "intermediate"
        else:
            print(
                "Error: Unknow automaton state. Cannot determine if it is done or not.",
                file=sys.stderr,
            )
            sys.exit(1)

        if self.red_agent == self.blue_flag or self.blue_agent == self.red_flag:
            done = True
        else:
            pass

        # Construct diagonostic info dict
        locations: MapLocations = MapLocations(
            self.blue_agent,
            self.red_agent,
            self.blue_flag,
            self.red_flag,
            self.blue_territory,
            self.red_territory,
            self.obstacles,
            self.walls,
        )
        all_info: dict = {
            "rhos": rhos_next,
            "aut_state": self.aut_state,
            "last_successful_aut_state": self.last_successful_aut_state,
            "locations": locations,
            "aut_state_status": status,
        }

        return (obs_next, reward, done, all_info)

    def save_info(self) -> dict:
        locations: MapLocations = MapLocations(
            self.blue_agent,
            self.red_agent,
            self.blue_flag,
            self.red_flag,
            self.blue_territory,
            self.red_territory,
            self.obstacles,
            self.walls,
        )
        info: dict = {
            "aut_state": self.aut_state,
            "locations": locations,
        }

        return info

    def save_action_prob(self, action_prob: list[float]):
        self.action_probs.append(action_prob)

    def save_action(self, action: int):
        self.actions.append(action)

    def render(self, mode: str = "console"):
        if mode != "console":
            raise NotImplementedError()

        _, ax = plt.subplots()
        h, w = self.init_map.shape

        markersize = 24
        ax.plot(
            self.blue_flag.x,
            self.blue_flag.y,
            marker=">",
            color="b",
            markersize=markersize,
        )
        ax.plot(
            self.red_flag.x,
            self.red_flag.y,
            marker=">",
            color="firebrick",
            markersize=markersize,
        )
        ax.set_xlim(0, w - 0.5)
        ax.set_ylim(0, h - 0.5)
        ax.set_aspect(1)
        ax.invert_yaxis()

        for obs in self.obstacles:
            obs_rec = Rectangle((obs.x, obs.y), 1, 1, color="black")
            ax.add_patch(obs_rec)

        for bt in self.blue_territory:
            bt_rec = Rectangle((bt.x, bt.y), 1, 1, color="cornflowerblue")
            ax.add_patch(bt_rec)

        for rt in self.red_territory:
            rt_rec = Rectangle((rt.x, rt.y), 1, 1, color="lightcoral")
            ax.add_patch(rt_rec)

        blue_agent_fight_region = Circle(
            (self.blue_agent.x, self.blue_agent.y),
            2,
            fill=False,
            ec="grey",
            label="Fight region",
        )
        ax.add_artist(blue_agent_fight_region)
        ax.plot(
            self.blue_agent.x,
            (self.blue_agent.y),
            marker="o",
            color="b",
            markersize=markersize,
            label="Blue agent",
        )
        ax.plot(
            self.red_agent.x,
            (self.red_agent.y),
            marker="o",
            color="firebrick",
            markersize=markersize,
        )
        plt.show(block=False)


def create_env(
    tl_spec: str,
    policy: Union[KnownPolicy, UnknownPolicy],
    obs_props_all: list[ObsProp],
    atom_prop_dict_all: dict[str, str],
    enemy_policy: EnemyPolicyMode,
    is_initialized_both_territories: bool,
) -> Environment:
    prop_vars = get_vars(tokenize(tl_spec), atom_prop_dict_all)
    used_obs = get_used_obs(prop_vars, atom_prop_dict_all, obs_props_all)
    n_in: int = len(used_obs)
    atom_prop_dict: dict[str, str] = {
        prop_var: atom_prop_dict_all[prop_var] for prop_var in prop_vars
    }
    obs_props: list[ObsProp] = [
        obs_prop for obs_prop in obs_props_all if obs_prop.name in used_obs
    ]
    aut = TLAutomaton(tl_spec, atom_prop_dict, obs_props)
    map_props = MapProps(policy.map, policy.fixed_map_locs, policy.map_object_ids)
    env_props = EnvProps(tl_spec, obs_props, atom_prop_dict, enemy_policy, map_props)
    env = Environment(
        env_props,
        is_initialized_both_territories,
    )

    return env


# def restore_model_tmp(
#     model_name: str,
#     spec: str,
#     obs_props_all: list[ObsProp],
#     atom_prop_dict_all: dict[str, str],
#     map: NDArray,
#     map_object_ids: MapObjectIDs,
#     enemy_policy: EnemyPolicyMode,
#     gpu: int,
#     is_initialized_in_both_territories: bool,
# ) -> PPO:

#     prop_vars = get_vars(tokenize(spec), atom_prop_dict_all)
#     used_obs = get_used_obs(prop_vars, atom_prop_dict_all, obs_props_all)
#     n_in: int = len(used_obs)
#     atom_prop_dict: dict[str, str] = {
#         prop_var: atom_prop_dict_all[prop_var] for prop_var in prop_vars
#     }
#     obs_props: list[ObsProp] = [
#         obs_prop for obs_prop in obs_props_all if obs_prop.name in used_obs
#     ]
#     aut = TLAutomaton(spec, atom_prop_dict, obs_props)

#     env = Environment(
#         n_in, map, map_object_ids, aut, enemy_policy, is_initialized_in_both_territories
#     )
#     env_monitored = Monitor(env)
#     model = PPO.load(
#         model_name, env_monitored, device=torch.device("cuda:{}".format(gpu))
#     )

#     return cast(PPO, model)


def restore_model(
    model_props: ModelProps,
    obs_props: list[ObsProp],
    device: torch.device,
    is_initialized_in_both_territories: bool,
) -> PPO | DQN:
    env = Environment(
        EnvProps(
            model_props.spec,
            obs_props,
            model_props.atom_prop_dict,
            model_props.enemy_policy,
            model_props.map_props,
        ),
        is_initialized_in_both_territories,
    )
    env_monitored = Monitor(env)
    model: PPO | DQN = (
        cast(
            PPO,
            PPO.load(
                model_props.model_name,
                env_monitored,
                device=device,
            ),
        )
        if model_props.rl_algorithm == "ppo"
        else cast(DQN, DQN.load(model_props.model_name, env_monitored, device=device))
    )

    return model


def act(loc: Location, action: Action) -> Location:
    y, x = loc
    new_loc: Location = loc

    if action == 0:  # stay
        pass

    elif action == 1:  # left
        new_loc = Location(y, x - 1)
    elif action == 2:  # down
        new_loc = Location(y + 1, x)
    elif action == 3:  # right
        new_loc = Location(y, x + 1)
    elif action == 4:  # up
        new_loc = Location(y - 1, x)
    else:
        print(
            "Error: the action should be int of either 0, 1, 2, 3, or 4.",
            file=sys.stderr,
        )
        sys.exit(1)

    return new_loc

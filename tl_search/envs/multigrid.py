from itertools import chain
import random
from random import Random
from abc import abstractmethod, ABC
from typing import Final, Literal


import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns

from tl_search.common.typing import Location
from tl_search.envs.typing import FieldObj, FixedObj, InfoDict, ObjectId, ObsDict, Path
from tl_search.map.utils import distance_area_point, distance_points, tuples2locs

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


class BaseMultigrid(gym.Env, ABC):
    def __init__(
        self,
        map_path: str,
        observation_space: spaces.Dict | None = None,
        is_move_clipped: bool = True,
        randomness: float = 0.75,
        battle_reward_alpha: float = 0.25,
        obstacle_penalty_beta: float | None = None,
        step_penalty_gamma: float = 0.0,
        capture_reward: float = 1.0,
        num_max_steps: int = 500,
        render_mode: Literal["human", "rgb_array"] = "human",
    ) -> None:
        super().__init__()

        self._map_path: Final[str] = map_path
        self._is_move_clipped: Final[bool] = is_move_clipped
        self._randomness: Final[float] = randomness
        self._battle_reward_alpha: Final[float] = battle_reward_alpha
        self._obstacle_penalty_beta: Final[float | None] = obstacle_penalty_beta
        self._step_penalty_gamma: Final[float] = step_penalty_gamma
        self._capture_reward: Final[float] = capture_reward
        self._num_max_steps: Final[int] = num_max_steps
        self._render_mode: Final[Literal["human", "rgb_array"]] = render_mode

        self._field_map: Final[NDArray] = np.loadtxt(map_path)

        # 0: stay, 1: left, 2: down, 3: right, 4: up
        self._moves: tuple[Location, ...] = (
            Location(0, 0),
            Location(0, -1),
            Location(-1, 0),
            Location(0, 1),
            Location(1, 0),
        )

        self.action_space = spaces.Discrete(len(self._moves))

        self._object_id = ObjectId(0, 1, 2, 3, 4, 5, 6, 7, 8)

        obstacle: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id.obstacle)))  # type: ignore
        )
        blue_flag: Final[Location] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id.blue_flag)))  # type: ignore
        )[0]

        red_flag: Final[Location] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id.red_flag)))  # type: ignore
        )[0]
        blue_background: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id.blue_background)))  # type: ignore
        ) + [blue_flag]
        red_background: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id.red_background)))  # type: ignore
        ) + [red_flag]

        h, w = self._field_map.shape
        if w % 2 != 0:
            raise Exception("[tl_search] The map shape should be in odds.")
        else:
            pass

        wall: Final[list[Location]] = list(
            set(
                [Location(-1, i) for i in range(-1, w + 1)]
                + [Location(h, i) for i in range(-1, w + 1)]
                + [Location(i, -1) for i in range(-1, h + 1)]
                + [Location(i, w) for i in range(-1, h + 1)]
            )
        )

        self._fixed_obj: Final[FixedObj] = FixedObj(
            blue_background, red_background, blue_flag, red_flag, obstacle, wall
        )

        self.observation_space: Final = (
            spaces.Dict(
                {
                    "blue_agent": spaces.Box(
                        low=np.array([-1, -1]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                    ),
                    "red_agent": spaces.Box(
                        low=np.array([-1, -1]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                    ),
                    "blue_flag": spaces.Box(
                        low=np.array([0, 0]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                    ),
                    "red_flag": spaces.Box(
                        low=np.array([0, 0]), high=np.array(self._field_map.shape) - 1, dtype=int  # type: ignore
                    ),
                    "blue_background": spaces.Box(
                        low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(blue_background))]))),  # type: ignore
                        high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(blue_background))]))).flatten() - 1,  # type: ignore
                        dtype=int,  # type: ignore
                    ),
                    "red_background": spaces.Box(
                        low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(red_background))]))),  # type: ignore
                        high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(red_background))]))).flatten() - 1,  # type: ignore
                        dtype=int,  # type: ignore
                    ),
                    "obstacle": spaces.Box(
                        low=np.array(list(chain.from_iterable([[0, 0] for _ in range(len(obstacle))]))),  # type: ignore
                        high=np.array(list(chain.from_iterable([self._field_map.shape for _ in range(len(obstacle))]))).flatten() - 1,  # type: ignore
                        dtype=int,  # type: ignore
                    ),
                    "is_red_agent_defeated": spaces.Discrete(2),
                }
            )
            if observation_space is None
            else observation_space
        )

    def _get_obs(self) -> ObsDict:
        observation: ObsDict = {
            "blue_agent": np.array(self._blue_agent_loc),
            "red_agent": np.array(self._red_agent_loc),
            "blue_flag": np.array(self._fixed_obj.blue_flag).flatten(),
            "red_flag": np.array(self._fixed_obj.red_flag).flatten(),
            "blue_background": np.array(self._fixed_obj.blue_background).flatten(),
            "red_background": np.array(self._fixed_obj.red_background).flatten(),
            "obstacle": np.array(self._fixed_obj.obstacle).flatten(),
            "is_red_agent_defeated": np.array(int(self._is_red_agent_defeated)),
        }
        return observation

    def _get_info(self) -> InfoDict:
        info = {
            "d_ba_ra": distance_points(self._blue_agent_loc, self._red_agent_loc),
            "d_ba_bf": distance_points(self._blue_agent_loc, self._fixed_obj.blue_flag),
            "d_ba_rf": distance_points(self._blue_agent_loc, self._fixed_obj.red_flag),
            "d_ra_bf": distance_points(self._red_agent_loc, self._fixed_obj.blue_flag),
            "d_ra_rf": distance_points(self._red_agent_loc, self._fixed_obj.red_flag),
            "d_bf_rf": distance_points(
                self._fixed_obj.blue_flag, self._fixed_obj.red_flag
            ),
            "d_ba_bb": distance_area_point(
                self._blue_agent_loc, self._fixed_obj.blue_background
            ),
            "d_ba_rb": distance_area_point(
                self._blue_agent_loc, self._fixed_obj.red_background
            ),
            "d_ra_bb": distance_area_point(
                self._red_agent_loc, self._fixed_obj.blue_background
            ),
            "d_ra_rb": distance_area_point(
                self._red_agent_loc, self._fixed_obj.red_background
            ),
            "d_ba_ob": distance_area_point(
                self._blue_agent_loc, self._fixed_obj.obstacle
            ),
        }
        return info

    def reset(
        self,
        blue_agent_loc: Location | None = None,
        red_agent_loc: Location | None = None,
        is_red_agent_defeated: bool = False,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[ObsDict, InfoDict]:
        super().reset(seed=seed, options=options)
        self._blue_agent_loc = (
            blue_agent_loc
            if blue_agent_loc is not None
            else Random().choice(self._fixed_obj.blue_background)
        )
        self._red_agent_loc = (
            red_agent_loc
            if red_agent_loc is not None
            else Random().choice(self._fixed_obj.red_background)
        )

        self.blue_traj = [self._blue_agent_loc]
        self.red_traj = [self._red_agent_loc]

        self._is_red_agent_defeated: bool = is_red_agent_defeated

        self._step_count: int = 0
        self._episodic_reward: float = 0.0

        self._field = FieldObj(
            self._blue_agent_loc, self._red_agent_loc, *self._fixed_obj
        )

        observation: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list: list[ObsDict] = [observation]

        return observation, info

    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]:
        if self._is_red_agent_defeated:
            pass
        else:
            self._red_agent_loc = self._enemy_act()

        self._blue_agent_loc = self._act(self._blue_agent_loc, action)

        self.blue_traj.append(self._blue_agent_loc)
        self.red_traj.append(self._red_agent_loc)

        reward, terminated, truncated = self._reward()

        self._step_count += 1
        self._episodic_reward += reward

        if self._num_max_steps <= self._step_count:
            truncated = True
        else:
            pass

        observation: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list.append(observation)

        return observation, reward, terminated, truncated, info

    def _act(self, loc: Location, action: int) -> Location:
        direction = self._moves[action]
        new_loc = Location(loc.y + direction.y, loc.x + direction.x)
        match self._is_move_clipped:
            case True:
                num_row: int
                num_col: int
                num_row, num_col = self._field_map.shape
                new_loc = Location(
                    np.clip(new_loc.y, 0, num_row - 1),
                    np.clip(new_loc.x, 0, num_col - 1),
                )

                if new_loc in self._fixed_obj.obstacle:
                    pass
                else:
                    loc = new_loc
            case False:
                loc = new_loc
        return loc

    @abstractmethod
    def _enemy_act(self) -> Location:
        ...

    def _reward(self) -> tuple[float, bool, bool]:
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = self._step_count >= self._num_max_steps

        if self._blue_agent_loc == self._fixed_obj.red_flag:
            reward += 1.0
            terminated = True
        else:
            pass

        if self._red_agent_loc == self._fixed_obj.blue_flag:
            reward -= 1.0
            terminated = True
        else:
            pass

        if (
            distance_points(self._blue_agent_loc, self._red_agent_loc) <= 1
            and not self._is_red_agent_defeated
        ):
            blue_win: bool

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
                reward += self._battle_reward_alpha * self._capture_reward
                self._is_red_agent_defeated = True
            else:
                reward -= self._battle_reward_alpha * self._capture_reward
                terminated = True

        if self._obstacle_penalty_beta is not None:
            if self._blue_agent_loc in self._fixed_obj.obstacle:
                reward -= self._obstacle_penalty_beta * self._capture_reward
                terminated = True
            else:
                pass
        else:
            pass

        reward -= self._step_penalty_gamma * self._capture_reward

        return reward, terminated, truncated

    def get_agent_trajs(self) -> tuple[Path, Path]:
        return self.blue_traj, self.red_traj

    def render(
        self,
        figure_save_path: str | None = None,
        markersize: int = 24,
        is_gui_shown: bool = False,
    ) -> NDArray | None:
        fig, ax = self._render_fixed_objects(markersize)

        ax.plot(
            self._blue_agent_loc.x,
            self._blue_agent_loc.y,
            marker="o",
            color="royalblue",
            markersize=markersize,
        )
        ax.plot(
            self._red_agent_loc.x,
            self._red_agent_loc.y,
            marker="o",
            color="crimson" if not self._is_red_agent_defeated else "lightgrey",
            markersize=markersize,
        )

        if figure_save_path is not None:
            fig.savefig(figure_save_path, dpi=600)
            plt.close()
        else:
            pass

        if is_gui_shown:
            plt.show()
            plt.close()
        else:
            pass

        output: NDArray | None

        match self._render_mode:
            case "rgb_array":
                fig.canvas.draw()
                output = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            case "human":
                output = None
            case _:
                raise Exception("[tl_search] The render mode is not defined.")

        return output

    def render_animation(
        self,
        path: str,
        obs_list: list[ObsDict] | None = None,
        marker_size: int = 24,
        interval: int = 200,
        is_gui_shown: bool = False,
    ) -> None:
        fig, ax = self._render_fixed_objects()

        artists: list[list[Line2D]] = []

        obs_list = self.obs_list if obs_list is None else obs_list

        blue_path: list[Location] = []
        red_path: list[Location] = []
        red_agent_status_traj: list[bool] = []

        for obs in obs_list:
            blue_path.append(obs["blue_agent"].tolist())
            red_path.append(obs["red_agent"].tolist())
            red_agent_status_traj.append(bool(obs["is_red_agent_defeated"]))

        for blue, red, is_red_agent_defeated in zip(
            blue_path, red_path, red_agent_status_traj
        ):
            blue_artist = ax.plot(
                blue[1], blue[0], marker="o", color="royalblue", markersize=marker_size
            )
            red_artist = ax.plot(
                red[1],
                red[0],
                marker="o",
                color="crimson" if not is_red_agent_defeated else "lightgrey",
                markersize=marker_size,
            )
            artists.append(blue_artist + red_artist)

        anim = ArtistAnimation(fig, artists, interval=interval)
        anim.save(path)

        if is_gui_shown:
            plt.show()
        else:
            pass

        plt.clf()
        plt.close()

    def _render_fixed_objects(
        self, markersize: int = 30, is_gui_shown: bool = False
    ) -> tuple[Figure, Axes]:
        sns.reset_defaults()
        plt.rcParams["font.family"] = "Times New Roman"
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        h, w = self._field_map.shape

        blue_flag: Location
        red_flag: Location
        obstacle: list[Location]
        (
            blue_background,
            red_background,
            blue_flag,
            red_flag,
            obstacle,
            _,
            _,
            _,
        ) = self._fixed_obj
        ax.set_xlim(-0.5, w - 1)
        ax.set_ylim(-0.5, h - 1)
        ax.set_aspect(1)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0.5, w + 0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, h + 0.5, 1), minor=True)
        ax.set_xticks(np.arange(0, w, 1))
        ax.set_yticks(np.arange(0, h, 1))
        ax.grid(which="minor")
        ax.tick_params(which="minor", length=0)

        for obs in obstacle:
            obs_rec = Rectangle((obs.x - 0.5, obs.y - 0.5), 1, 1, color="black")
            ax.add_patch(obs_rec)

        for bb in blue_background:
            bb_rec = Rectangle((bb.x - 0.5, bb.y - 0.5), 1, 1, color="aliceblue")
            ax.add_patch(bb_rec)

        for rb in red_background:
            rf_rec = Rectangle((rb.x - 0.5, rb.y - 0.5), 1, 1, color="mistyrose")
            ax.add_patch(rf_rec)

        ax.plot(
            blue_flag.x,
            blue_flag.y,
            marker=">",
            color="mediumblue",
            markersize=markersize,
        )
        ax.plot(
            red_flag.x,
            red_flag.y,
            marker=">",
            color="firebrick",
            markersize=markersize,
        )

        if is_gui_shown:
            plt.show()
            plt.close()
        else:
            pass

        return fig, ax

    def seed(self, seed: int) -> None:
        self._seed = seed

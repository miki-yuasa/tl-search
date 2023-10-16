from typing import TypeAlias, NamedTuple, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from tl_search.common.typing import Location


class FieldObj(NamedTuple):
    blue_agent: Location
    red_agent: Location
    blue_background: list[Location]
    red_background: list[Location]
    blue_flag: Location
    red_flag: Location
    obstacle: list[Location]
    wall: list[Location]
    blue_border: list[Location] = []
    red_border: list[Location] = []
    is_red_agent_defeated: bool = False


class FixedObj(NamedTuple):
    blue_background: list[Location]
    red_background: list[Location]
    blue_flag: Location
    red_flag: Location
    obstacle: list[Location]
    wall: list[Location]
    blue_border: list[Location] = []
    red_border: list[Location] = []


class ObjectId(NamedTuple):
    blue_background: int
    red_background: int
    blue_ugv: int
    blue_uav: int
    red_ugv: int
    red_uav: int
    blue_flag: int
    red_flag: int
    obstacle: int


ObsDict: TypeAlias = dict[str, NDArray]
InfoDict: TypeAlias = dict[str, float]

Path: TypeAlias = list[Location]

EnemyPolicyMode: TypeAlias = Literal["fight", "patrol", "capture", "none"]


@dataclass_json
@dataclass
class HeuristicEnemyEnvTrainingConfig:
    enemy_policy_mode: EnemyPolicyMode
    seeds: list[int]
    num_envs: int
    total_time_steps: int
    map_path: str
    model_save_path: str
    learning_curve_path: str
    animation_save_path: str
    window: int
    tuned_param_name: str | None = None
    tuned_param_values: list[float] | None = None


class TLMultigridTrainingConfig(TypedDict, total=False):
    tl_spec: str
    enemy_policy_mode: EnemyPolicyMode
    seeds: list[int]
    num_envs: int
    total_timesteps: int
    map_path: str
    model_save_path: str
    learning_curve_path: str
    animation_save_path: str
    window: int
    tuned_param_name: str | None
    tuned_param_values: list[float] | None


class ModelStats(TypedDict):
    capture_defeat_count: int
    capture_count: int
    defeat_count: int
    lose_count: int
    reward_mean: float | np.floating
    reward_std: float | np.floating
    timesteps_mean: float | np.floating
    timesteps_std: float | np.floating
    rewards: list[float]
    timesteps: list[int]

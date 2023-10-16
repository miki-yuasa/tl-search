from typing import NamedTuple, Callable, TypedDict, Union, Literal

import numpy as np
from numpy.typing import NDArray


SaveMode = Literal["enabled", "suppressed", "disabled"]
PolicyMode = Literal["known", "unknown"]
FilterMode = Literal["mean", "elementwise"]
RankOption = Literal["kl_div", "reward"]
AutomatonStateStatus = Literal["intermediate", "trap", "goal"]
Combination = Literal[0, 1, 2]  # i.e. 0: (|) & (|), 1: (&) & (|), 2: (|) & (&)
Exclusion = Literal["negation", "group"]
EnemyPolicyMode = Literal["random", "a-star"]

CtFMap = NDArray

Machines = Literal["ti", "dl"]
RLAlgorithm = Literal["ppo", "dqn"]


class Robustness(NamedTuple):
    name: str
    value: float


class RobustnessCounter(NamedTuple):
    robustness: float
    ind: int


class AutomatonStateCounter(NamedTuple):
    state: int
    ind: int


class Transition(NamedTuple):
    condition: str
    next_state: int
    is_trapped_next: bool = False


class SymbolProp(NamedTuple):
    priority: int
    func: Callable


class AtomicPropositions(NamedTuple):
    num_APs: int
    APs: list[str]


class ObsProp(NamedTuple):
    name: str
    args: list[str]
    func: Callable


class Location(NamedTuple):
    y: int
    x: int


class AStarNode(NamedTuple):
    f: int
    g: int
    h: int
    parent: Union["AStarNode", None]
    loc: Location


class FixedMapLocations(NamedTuple):
    blue_flag: Location
    red_flag: Location
    blue_territory: list[Location]
    red_territory: list[Location]
    obstacles: list[Location]
    walls: list[Location]


class MapLocations(NamedTuple):
    blue_agent: Location
    red_agent: Location
    blue_flag: Location
    red_flag: Location
    blue_territory: list[Location]
    red_territory: list[Location]
    obstacles: list[Location]
    walls: list[Location]


class MapObjectIDs(NamedTuple):
    blue_agent: int
    red_agent: int
    blue_flag: int
    red_flag: int
    blue_territory: Union[int, None]
    red_territory: Union[int, None]
    obstacles: int


Action = Literal[-1, 0, 1, 2, 3, 4]

ActionProb = tuple[float, float, float, float, float]
KLDivsSpec = list[list[float]]
ActionProbsSpec = list[list[ActionProb]]


class SpecLogger(NamedTuple):
    tl_spec: str
    kl_divs: KLDivsSpec
    tl_action_probs: ActionProbsSpec
    original_action_probs: ActionProbsSpec


class KLDivLogger(NamedTuple):
    min_tl_spec: str
    min_kl_div: float
    mean_kl_div_mean: float
    max_kl_div_mean: float
    tl_specs: list[str]
    kl_divs: list[float]
    tl_action_probs: list[ActionProbsSpec]
    ori_action_probs: list[ActionProbsSpec]


class SortedLogger(NamedTuple):
    experiment: int | str
    sorted_specs: list[str]
    sorted_min_kl_divs: list[float]


class SortedLog(TypedDict):
    run: str
    specs: list[str]
    kl_divs: list[float]


class FilteredLog(TypedDict):
    run: str
    specs: list[str]
    rewards: list[float]
    kl_divs: list[float]
    episode_lengths: list[float]


class MultiStartSearchLogger(NamedTuple):
    experiment: int
    sorted_local_optimal_specs: list[str]
    sorted_local_optimal_kl_divs: list[float]
    searches_traces: list[tuple[list[str], list[float]]]


class MultiStartSearchLog(TypedDict):
    run: int
    sorted_local_optimal_specs: list[str]
    sorted_local_optimal_kl_divs: list[float]
    total_searched_specs: int
    nums_searched_specs: list[int]
    searches_traces: list[tuple[list[str], list[float]]]


class ExperimentLogger(NamedTuple):
    policy_mode: str
    experiment: int | str
    description: str
    min_kl_div_spec: str
    max_reward_spec: str
    min_kl_div_mean: float
    mean_kl_div_mean: float
    max_kl_div_mean: float
    min_reward_mean: float
    mean_reward_mean: float
    max_reward_mean: float
    tl_specs: list[str]
    kl_divs: list[list[float]]
    tl_action_probs: list[ActionProbsSpec]
    ori_action_probs: list[ActionProbsSpec]
    model_names: list[list[str]]
    mean_rewards: list[list[float]]
    std_rewards: list[list[float]]


class MultiStepKLDivLogger(NamedTuple):
    policy_mode: PolicyMode
    experiment: int
    comparison_steps: int
    num_sampling: int
    min_kl_div_spec: str
    max_reward_spec: str
    min_kl_div_mean: float
    mean_kl_div_mean: float
    max_kl_div_mean: float
    min_reward_mean: float
    mean_reward_mean: float
    max_reward_mean: float
    tl_specs: list[str]
    kl_divs: list[float]
    mean_rewards: list[list[float]]


class AccuracyLogger(NamedTuple):
    max_tl_spec: str
    max_accuracy_mean: float
    mean_accuracy_mean: float
    min_accuracy_mean: float
    tl_specs: list[str]
    accuracies: list[float]


class ProbabilityLogger(NamedTuple):
    max_prob_spec: str
    max_prob_mean: float
    mean_prob_mean: float
    min_prob_mean: float
    tl_specs: list[str]
    probabilities: list[float]


class PositiveRewardLogger(NamedTuple):
    tl_specs_pos: list[str]
    mean_rewards_pos: list[float]
    kl_divs_pos: list[float]
    mean_rewards_pos_inds: list[int]
    model_names_pos: list[str]


class PositiveRewardLog(TypedDict):
    run: str
    specs: list[str]
    rewards: list[float]
    kl_divs: list[float]


class Hyperparameters(NamedTuple):
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float


class HyperparameterTuningLogger(NamedTuple):
    spec: str
    learning_rate: Union[float, list[float]]
    n_steps: Union[int, list[int]]
    batch_size: Union[int, list[int]]
    n_epochs: Union[int, list[int]]
    gamma: Union[float, list[float]]
    gae_lambda: Union[float, list[float]]
    clip_range: Union[float, list[float]]
    ent_coef: Union[float, list[float]]
    vf_coef: Union[float, list[float]]
    max_grad_norm: Union[float, list[float]]
    kl_divs: list[list[float]]


class ValueTable(NamedTuple):
    neg: list[bool]
    group: list[bool]
    temporal: list[bool]  # True for F, False for G


class SpecNode(NamedTuple):
    value_table: ValueTable
    F_star: bool  # False for |, True for &
    G_star: bool
    combination: Combination  # i.e. 0: (|) & (|), 1: (&) & (|), 2: (|) & (&)
    exclusions: list[Exclusion]
    vars: tuple[str, ...]


class MapProps(NamedTuple):
    map: NDArray
    fixed_map_locs: FixedMapLocations
    map_object_ids: MapObjectIDs


class ModelProps(NamedTuple):
    model_name: str
    rl_algorithm: RLAlgorithm
    spec: str
    atom_prop_dict: dict[str, str]
    enemy_policy: EnemyPolicyMode
    map_props: MapProps


class EnvProps(NamedTuple):
    spec: str
    obs_props: list[ObsProp]
    atom_prop_dict: dict[str, str]
    enemy_policy: EnemyPolicyMode
    map_props: MapProps


class KLDivReport(NamedTuple):
    kl_div_mean: float
    kl_div_std: float
    kl_div_max: float
    kl_div_min: float
    kl_ci: tuple[float, float]


class KLDivReportDict(TypedDict):
    kl_div_mean: float
    kl_div_std: float
    kl_div_max: float
    kl_div_min: float
    kl_ci: tuple[float, float]


class RewardReport(NamedTuple):
    mean: float
    std: float


class RewardReportDict(TypedDict):
    mean: float
    std: float


class ModelReport(NamedTuple):
    kl_div_report: KLDivReport
    reward_report: RewardReport


class SearchLog(TypedDict, total=False):
    run: str | int
    enemy_policy: str
    seeds: list[int]
    num_searched_specs: int
    min_kl_div_spec: str
    max_reward_spec: str
    min_kl_div: float
    mean_kl_div: float
    max_kl_div: float
    min_reward: float
    mean_reward: float
    max_reward: float
    searched_specs: list[str]
    kl_div_means: list[float]


class EpisodeLengthReport(TypedDict):
    mean: float
    std: float
    max: float
    min: float

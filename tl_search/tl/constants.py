from typing import Callable
from tl_search.policies.known_policy import KnownPolicy
from tl_search.policies.unknown_policy import UnknownPolicy

from tl_search.common.typing import Location, MapObjectIDs, ObsProp
from tl_search.map.utils import (
    distance_area_point,
    distance_points,
    get_middle_points,
)

NUM_SAMPLING: int = 1000
NUM_REPLICATES: int = 1
TOTAL_TIMESTEPS: int = 50_000
WRITE_PER_SPECS: int = 5

fight_range: float = 0.5
defense_range: float = 0.5
global_threshold: float = 0.9

known_policy_map_object_ids = MapObjectIDs(2, 4, 6, 7, 0, 1, 8)
unknown_policy_map_object_ids = MapObjectIDs(5, -5, 2, -2, None, None, 8)

choke_points: list[Location] = [
    Location(0, 4),
    Location(0, 5),
    Location(1, 4),
    Location(1, 5),
    Location(8, 4),
    Location(8, 5),
    Location(9, 4),
    Location(9, 5),
]

obs_info: list[tuple[str, list[str], Callable]] = [
    ("d_ba_ra", ["blue_agent", "red_agent"], lambda x, y: distance_points(x, y)),
    ("d_ba_rf", ["blue_agent", "red_flag"], lambda x, y: distance_points(x, y)),
    (
        "d_ba_rt",
        ["blue_agent", "red_territory"],
        lambda x, y: distance_area_point(x, y),
    ),
    ("d_ba_bf", ["blue_agent", "blue_flag"], lambda x, y: distance_points(x, y)),
    ("d_ra_bf", ["red_agent", "blue_flag"], lambda x, y: distance_points(x, y)),
    (
        "d_ra_bt",
        ["red_agent", "blue_territory"],
        lambda x, y: distance_area_point(x, y),
    ),
    (
        "d_ba_ob",
        ["blue_agent", "obstacles"],
        lambda x, y: distance_area_point(x, y),
    ),
    (
        "d_ba_wa",
        ["blue_agent"],
        lambda x: min(
            x[0] + 1,
            10 - x[0],
            x[1] + 1,
            10 - x[1],
        ),
    ),
    ("d_ba_uh", ["blue_agent"], lambda loc: 5 - loc.y),
    ("d_ba_lh", ["blue_agent"], lambda loc: loc.y - 4),
    ("d_ra_uh", ["red_agent"], lambda loc: 5 - loc.y),
    ("d_ra_lh", ["red_agent"], lambda loc: loc.y - 4),
    ("d_ba_cp", ["blue_agent"], lambda loc: distance_area_point(loc, choke_points)),
    (
        "d_ba_md",
        ["blue_agent", "red_agent", "blue_flag"],
        lambda ba, ra, bf: distance_area_point(ba, get_middle_points(ra, bf)),
    ),
    ("d_ba_ra_y", ["blue_agent", "red_agent"], lambda ba, ra: abs(ra.y - ba.y)),
]

atom_prop_dict_all: dict[str, str] = {
    "psi_ba_ra": "d_ba_ra < {}".format(global_threshold),
    "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
    "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
    "psi_ba_rt": "d_ba_rt < {}".format(global_threshold),
    "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
    "psi_ra_bt": "d_ra_bt < {}".format(global_threshold),
    "psi_ba_ob": "d_ba_ob < {}".format(global_threshold),
    "psi_ba_wa": "d_ba_wa < {}".format(global_threshold),
    "psi_ba_obs": "(d_ba_wa < {})|(d_ba_ob < {})".format(
        global_threshold, global_threshold
    ),
    "psi_ba_uh": "d_ba_uh < {}".format(global_threshold),
    "psi_ba_lh": "d_ba_lh < {}".format(global_threshold),
    "psi_ra_uh": "d_ra_uh < {}".format(global_threshold),
    "psi_ra_lh": "d_ra_lh < {}".format(global_threshold),
    "psi_ba_cp": "d_ba_cp < {}".format(global_threshold),
    "psi_ba_md": "d_ba_md < {}".format(global_threshold),
    "psi_ba_ra_y": "d_ba_ra_y < {}".format(global_threshold),
    "psi_bara_rabf": "(d_ba_ra - {}) < (d_ra_bf - {})".format(
        global_threshold, global_threshold
    ),
}

obs_props_all: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]

known_map_file = "maps/board_0002.txt"
unknown_policy_filename: str = "./tl_search/policies/data/CTF10_1v1_SubPolicy17.npz"

known_policy = KnownPolicy(known_map_file, known_policy_map_object_ids)
unknown_policy = UnknownPolicy(unknown_policy_filename, unknown_policy_map_object_ids)

"""
tl_specs = [
    "F psi_ba_ra & G!(psi_ra_bf | psi_ba_rt | psi_ba_ob | psi_ba_wa)",
    # "F psi_ba_ra & G((psi_ra_bt -> !psi_ra_bf) & !psi_ba_rt & !psi_ba_ob & !psi_ba_wa)",
    # "F psi_ba_ra & G((!psi_ra_bf) & (psi_ra_bt -> !psi_ba_rt) & !psi_ba_ob & !psi_ba_wa)",
    # "F psi_ba_ra & G((psi_ra_bt -> !psi_ra_bf) & (psi_ra_bt -> !psi_ba_rt) & !psi_ba_ob & !psi_ba_wa)",
]
"""

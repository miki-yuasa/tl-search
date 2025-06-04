import itertools
from numpy.typing import NDArray

from tl_search.common.typing import FixedMapLocations, Location, MapLocations
from tl_search.map.utils import parse_map, read_map
from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.tl.tl_parser import get_used_tl_props, tl2rob

spec = "F(psi_ba_ra) & G(psi_ba_bf & psi_ba_rf & psi_ba_obs)"
map_filename: str = "maps/board_0002.txt"
map: NDArray = read_map(map_filename)
fixed_map_locs: FixedMapLocations = parse_map(map, known_policy_map_object_ids)

all_locs: list[Location] = [
    Location(*pair) for pair in itertools.product(range(10), range(10))
]

loc_combinations: list[tuple[Location, Location]] = list(
    itertools.product(all_locs, all_locs)
)

states: list[tuple[float, float, float, float]] = []

for blue_agent, red_agent in loc_combinations:
    locations = MapLocations(blue_agent, red_agent, *fixed_map_locs)

    location_dict: dict[str, float] = locations._asdict()

    obs_props, atom_prop_dict = get_used_tl_props(
        spec, atom_prop_dict_all, obs_props_all
    )
    obs_dict: dict[str, float] = {
        obs.name: obs.func(*[location_dict[arg] for arg in obs.args])
        for obs in obs_props
    }
    atom_rob_dict: dict[str, float] = {
        atom_props_key: tl2rob(atom_prop_dict[atom_props_key], obs_dict)
        for atom_props_key in atom_prop_dict.keys()
    }

    states.append(tuple(atom_rob_dict.values()))

no_overlap_states: list[tuple[float, float, float, float]] = list(set(states))

print(f"Number of states: {len(states)}")
print(f"Number of states with no overlap: {len(no_overlap_states)}")

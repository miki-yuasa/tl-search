import os, random
import traceback
import math
from typing import Iterable, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

from tl_search.common.typing import (
    Location,
    FixedMapLocations,
    MapLocations,
    MapObjectIDs,
)


def map2props(map_filename: str, map_object_ids: MapObjectIDs):
    map: NDArray = read_map(map_filename)
    fixed_map_locs: FixedMapLocations = cast(
        FixedMapLocations, parse_map(map, map_object_ids)
    )

    return map, fixed_map_locs, map_object_ids


def read_map(filename: str) -> NDArray:
    """Read the map from a text file"""
    filepath = os.path.join(os.path.dirname(__file__), *os.path.split(filename))
    map: NDArray = np.loadtxt(filepath)
    return map


@overload
def parse_map(
    map: NDArray, map_object_ids: MapObjectIDs, parse_agent_locs: bool = False
) -> FixedMapLocations:
    ...


@overload
def parse_map(
    map: NDArray, map_object_ids: MapObjectIDs, parse_agent_locs: bool = True
) -> MapLocations:
    ...


def parse_map(
    map: NDArray, map_object_ids: MapObjectIDs, parse_agent_locs: bool = False
) -> Union[MapLocations, FixedMapLocations, None]:
    """Read map objects from the map"""

    blue_flag: Location = Location(
        *list(zip(*np.where(map == map_object_ids.blue_flag)))[0]
    )
    red_flag: Location = Location(
        *list(zip(*np.where(map == map_object_ids.red_flag)))[0]
    )
    obstacles: list[Location] = tuples2locs(
        list(
            cast(
                Iterable[tuple[int, int]],
                zip(*np.where(map == map_object_ids.obstacles)),
            )
        )
    )

    h, w = map.shape
    if w % 2 != 0:
        try:
            raise Exception("The map shape should be in odds.")
        except Exception as e:
            print(e)
            traceback.print_exc()
    else:
        pass

    walls: list[Location] = list(
        set(
            [Location(-1, i) for i in range(-1, w + 1)]
            + [Location(h, i) for i in range(-1, w + 1)]
            + [Location(i, -1) for i in range(-1, h + 1)]
            + [Location(i, w) for i in range(-1, h + 1)]
        )
    )

    blue_territory: list[Location]
    red_territory: list[Location]

    if map_object_ids.blue_territory != None and map_object_ids.red_territory != None:
        blue_territory = tuples2locs(
            list(
                cast(
                    Iterable[tuple[int, int]],
                    zip(
                        *np.where(
                            (map == map_object_ids.blue_territory)
                            | (map == map_object_ids.blue_agent)
                            | (map == map_object_ids.blue_flag)
                        )
                    ),
                )
            )
        )
        red_territory = tuples2locs(
            list(
                cast(
                    Iterable[tuple[int, int]],
                    zip(
                        *np.where(
                            (map == map_object_ids.red_territory)
                            | (map == map_object_ids.red_flag)
                            | (map == map_object_ids.red_agent)
                        )
                    ),
                )
            )
        )
    else:
        # Harcode the territories
        # print(
        #    "The map element IDs does not contain those for agent territories.\nSeparate the map into halves on the vertical center line as their territories."
        # )
        red_half: list[Location] = [
            Location(y, x) for x in range(int(w / 2)) for y in range(h)
        ]
        blue_half: list[Location] = [
            Location(y, x) for x in range(int(w / 2)) for y in range(h)
        ]

        red_territory = list(set(red_half) - set(obstacles))
        blue_territory = list(set(blue_half) - set(obstacles))

    all_locations: Union[MapLocations, FixedMapLocations]

    if parse_agent_locs:
        try:
            blue_agent: Location = Location(
                *list(zip(*np.where(map == map_object_ids.blue_agent)))[0]
            )
        except:
            return None
        try:
            red_agent: Location = Location(
                *list(zip(*np.where(map == map_object_ids.red_agent)))[0]
            )
        except:
            return None
        all_locations = MapLocations(
            blue_agent,
            red_agent,
            blue_flag,
            red_flag,
            blue_territory,
            red_territory,
            obstacles,
            walls,
        )
    else:
        all_locations = FixedMapLocations(
            blue_flag, red_flag, blue_territory, red_territory, obstacles, walls
        )

    return all_locations


def sample_agent_locs(
    fixed_locs: FixedMapLocations, is_from_all_territories: bool
) -> MapLocations:
    blue_agent: Location = random.choice(
        (
            fixed_locs.blue_territory
            + fixed_locs.red_territory
            + fixed_locs.obstacles
            + fixed_locs.walls
            if is_from_all_territories
            else list(
                set(fixed_locs.blue_territory + fixed_locs.red_territory)
                - set([fixed_locs.red_flag])
            )
        )
    )

    red_agent: Location = random.choice(
        (
            fixed_locs.blue_territory + fixed_locs.red_territory
            if is_from_all_territories
            else list(
                set(fixed_locs.blue_territory + fixed_locs.red_territory)
                - set([fixed_locs.blue_flag])
            )
        )
    )

    return MapLocations(blue_agent, red_agent, *fixed_locs)


def index2loc(index: int, map_shape: tuple[int, int]) -> tuple[int, int]:
    """Convert a vector map index to its correpsonding zero-based location"""
    h, w = map_shape
    y, x = divmod(index, w)
    return (x, y)


def loc2index(location: tuple[int, int], map_shape: tuple[int, int]) -> int:
    """Convert a zero-based map location to its vector index"""
    h, w = map_shape
    x, y = location
    index = y * w + x
    return index


def tuples2locs(tuples: list[tuple[int, int]]) -> list[Location]:
    locs: list[Location] = [Location(*item) for item in tuples]
    return locs


def locs2tuples(locs: list[Location]) -> list[tuple[int, int]]:
    tuples: list[tuple[int, int]] = cast(
        list[tuple[int, int]], [tuple(item) for item in locs]
    )
    return tuples


def distance_points(p1: Location, p2: Location, is_defeated: bool = False) -> float:
    """Calculate the squared distance of two points"""
    return (
        float(np.linalg.norm(np.array(p1) - np.array(p2)))
        if not is_defeated
        else float("inf")
    )


def distance_area_point(point: Location, area: list[Location]) -> float:
    """Calculate the squared distance of an area and a point"""
    distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in area]
    return float(np.min(distances))


def closest_area_point(point: Location, area: list[Location]) -> Location:
    """Calculate the squared distance of an area and a point"""
    distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in area]
    return area[np.argmin(distances)]


def mirror_loc(loc: NDArray, map_shape: tuple[int, int]) -> NDArray:
    r, c = map_shape
    return np.array(loc[0], c - loc[1] - 1)


def mirror_flat_locs(flat_locs: NDArray, map_shape: tuple[int, int]) -> NDArray:
    r, c = map_shape

    locs = flat_locs.reshape([-1, 2])
    locs[:, 1] = c - locs[:, 1] - 1

    return locs.reshape([-1])


def get_middle_points(p1: Location, p2: Location) -> list[Location]:
    mid_x: float = (p1.x + p2.x) / 2
    mid_y: float = (p1.y + p2.y) / 2

    x_c: int = math.ceil(mid_x)
    x_f: int = math.floor(mid_x)
    y_c: int = math.ceil(mid_y)
    y_f: int = math.floor(mid_y)

    mid_pts: list[Location] = list(
        set(
            [
                Location(y_c, x_c),
                Location(y_c, x_f),
                Location(y_f, x_c),
                Location(y_f, x_f),
            ]
        )
    )

    return mid_pts


def get_neighbors(current_loc: Location) -> list[Location]:
    directions: list[Location] = [
        Location(0, 1),
        Location(0, -1),
        Location(1, 0),
        Location(-1, 0),
    ]

    neighbors: list[Location] = [
        Location(current_loc.y + direction.y, current_loc.x + direction.x)
        for direction in directions
    ] + [current_loc]

    return neighbors


def get_allowed_locations(
    current_loc: Location, obstacles: list[Location]
) -> list[Location]:
    possible_locs: list[Location] = get_neighbors(current_loc)

    allowed_locs: list[Location] = list(set(possible_locs) - set(obstacles))

    return allowed_locs


def manhattan_distance(p1: Location, p2: Location) -> int:
    """
    Compute a Manhattan distance of two points

    Parameters:
    p1 (Location): a location
    p2 (Location): another location

    Returns:
    distance (int): Manhattan distance of the two points
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

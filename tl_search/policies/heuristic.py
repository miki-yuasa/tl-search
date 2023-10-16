from abc import ABC, abstractmethod
from heapq import heapify, heappop, heappush
import random

import numpy as np
from numpy.typing import NDArray

from tl_search.map.utils import (
    get_allowed_locations,
    manhattan_distance,
)
from tl_search.common.typing import AStarNode, FixedMapLocations, Location


def a_star(start: Location, end: Location, map: NDArray) -> list[Location]:
    """
    Compute the path-planning for the red agent using A* algorithm

    Parameters:
    start (Location): start location
    end (Location): goal location

    Returns:
    path (list[Location]): the path from its original locition to the goal.
    """
    rows, cols = map.shape
    map_list: list[list[float]] = map.tolist()
    # Add the start and end nodes
    start_node = AStarNode(
        manhattan_distance(start, end), 0, manhattan_distance(start, end), None, start
    )
    # Initialize and heapify the lists
    open_nodes: list[AStarNode] = [start_node]
    closed_nodes: list[AStarNode] = []
    heapify(open_nodes)
    path: list[Location] = []  # return of the func

    while open_nodes:
        # Get the current node popped from the open list
        current_node = heappop(open_nodes)

        # Push the current node to the closed list
        closed_nodes.append(current_node)

        # When the goal is found
        if current_node.loc == end:
            current: AStarNode | None = current_node
            while current is not None:
                path.append(current.loc)
                current = current.parent

            path.reverse()
            break

        else:
            for direction in [
                Location(0, 1),
                Location(0, -1),
                Location(1, 0),
                Location(-1, 0),
            ]:
                # Get node location
                current_loc: Location = current_node.loc
                new_loc = Location(
                    current_loc.y + direction.y, current_loc.x + direction.x
                )

                # Make sure within a range
                if (
                    (new_loc.x >= 0 and new_loc.x < cols)
                    and (new_loc.y >= 0 and new_loc.y < rows)
                    and (map_list[new_loc.y][new_loc.x] != 8)
                ):
                    # Create the f, g, and h values
                    g = current_node.g + 1
                    h = manhattan_distance(new_loc, end)
                    f = g + h

                    # Check if the new node is in the open or closed list
                    open_indices = [
                        i
                        for i, open_node in enumerate(open_nodes)
                        if open_node.loc == new_loc
                    ]
                    closed_indices = [
                        i
                        for i, closed_node in enumerate(closed_nodes)
                        if closed_node.loc == new_loc
                    ]

                    # Compare f values if the new node is already existing in either list
                    if closed_indices:
                        closed_index = closed_indices[0]
                        if f < closed_nodes[closed_index].f:
                            closed_nodes.pop(closed_index)
                            heappush(
                                open_nodes, AStarNode(f, g, h, current_node, new_loc)
                            )
                        else:
                            continue

                    elif open_indices:
                        open_index = open_indices[0]
                        if f < open_nodes[open_index].f:
                            open_nodes.pop(open_index)
                            open_nodes.append(AStarNode(f, g, h, current_node, new_loc))
                            heapify(open_nodes)
                        else:
                            continue

                    else:
                        heappush(open_nodes, AStarNode(f, g, h, current_node, new_loc))

                else:
                    continue

    return path


class BaseHeuristicPolicy(ABC):
    def __init__(self, init_loc: Location) -> None:
        super().__init__()
        self._counter: int = 0
        self._traj: list[Location] = [init_loc]
        self._init_loc: Location = init_loc
        self._current_loc: Location = init_loc

    @abstractmethod
    def act(self):
        ...

    @property
    def traj(self) -> list[Location]:
        return self._traj

    @property
    def init_loc(self) -> Location:
        return self._init_loc

    @property
    def current_loc(self) -> Location:
        return self._current_loc

    @property
    def counter(self) -> int:
        return self._counter


class AStarPolicy(BaseHeuristicPolicy):
    def __init__(self, init_loc: Location, goal: Location, map: NDArray) -> None:
        super().__init__(init_loc)
        self._goal: Location = goal
        self._map: NDArray = map

        self.__traj: list[Location] = a_star(init_loc, goal, map)

    def act(self) -> Location:
        self._counter += 1
        self._init_loc = self.__traj[self._counter]

        return self._init_loc


class RandomPolicy(BaseHeuristicPolicy):
    def __init__(self, init_loc: Location, fixed_map_locs: FixedMapLocations) -> None:
        super().__init__(init_loc)
        self._fixed_map_locs: FixedMapLocations = fixed_map_locs

    def act(self) -> Location:
        allowed_locs: list[Location] = get_allowed_locations(
            self._current_loc,
            self._fixed_map_locs.obstacles + self._fixed_map_locs.walls,
        )

        new_loc: Location = random.choice(allowed_locs)
        self._current_loc = new_loc
        self._counter += 1

        return new_loc

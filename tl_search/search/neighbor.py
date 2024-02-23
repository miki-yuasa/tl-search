from math import floor
import os
import pickle
import random
from itertools import combinations, product
from tokenize import group
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from tl_search.common.io import spec2title

from tl_search.common.typing import Combination, Exclusion, SpecNode, ValueTable
from tl_search.envs.typing import EnemyPolicyMode
from tl_search.search.filter import is_nonsensical_spec


def initialize_node(
    vars: tuple[str, ...],
    exclusions: list[Exclusion],
) -> SpecNode:
    num_vars: int = len(vars)

    neg_row: list[bool] = (
        [False for _ in range(num_vars)]
        if "negation" in exclusions
        else [bool(random.randint(0, 1)) for _ in range(num_vars)]
    )
    group_row: list[bool] = (
        [False for _ in range(num_vars)]
        if "negation" in exclusions
        else [bool(random.randint(0, 1)) for _ in range(num_vars)]
    )

    temporal_row_tmp: list[bool] = [
        bool(random.randint(0, 1)) for _ in range(num_vars - 1)
    ]
    temporal_row_last_element: list[bool] = (
        [bool(random.randint(0, 1))]
        if list(set(temporal_row_tmp)) == 2
        else [not temporal_row_tmp[0]]
    )

    temporal_row: list[bool] = (
        [False for _ in range(num_vars)]
        if "negation" in exclusions
        else temporal_row_tmp + temporal_row_last_element
    )

    value_table = ValueTable(neg_row, group_row, temporal_row)

    node = SpecNode(
        value_table,
        bool(random.randint(0, 1)),
        bool(random.randint(0, 1)),
        cast(Literal[0, 1, 2], random.randint(0, 2)),
        exclusions,
        vars,
    )

    return node


def create_value_table(vars: tuple[str, ...]) -> NDArray:
    value_table: NDArray = np.ones((2, len(vars)))
    return value_table


def create_all_nodes(
    vars: tuple[str, ...], exclusions: list[Exclusion]
) -> list[SpecNode]:
    unique_exclusions: list[Exclusion] = list(set(exclusions))
    num_vars: int = len(vars)

    temporal_rows: list[list[bool]] = generate_value_row_vectors(
        num_vars, 1, num_vars - 1
    )
    group_rows: list[list[bool]] = (
        generate_value_row_vectors(num_vars, 0, floor(num_vars / 2))
        if "group" not in unique_exclusions
        else [[False for _ in range(num_vars)]]
    )
    neg_rows: list[list[bool]] = (
        generate_value_row_vectors(num_vars, 0, num_vars)
        if "negation" not in unique_exclusions
        else [[False for _ in range(num_vars)]]
    )

    value_tables: list[ValueTable] = [
        ValueTable(*couple) for couple in product(neg_rows, group_rows, temporal_rows)
    ]

    sign_combs: list[Combination] = [0, 1, 2]
    nodes: list[SpecNode] = []

    for sign_comb in sign_combs:
        nodes_tmp: list[SpecNode] = (
            [
                SpecNode(table, True, True, sign_comb, unique_exclusions, vars)
                for table in value_tables
            ]
            + [
                SpecNode(table, True, False, sign_comb, unique_exclusions, vars)
                for table in value_tables
            ]
            + [
                SpecNode(table, False, True, sign_comb, unique_exclusions, vars)
                for table in value_tables
            ]
            + [
                SpecNode(table, False, False, sign_comb, unique_exclusions, vars)
                for table in value_tables
            ]
        )
        nodes += nodes_tmp

        if num_vars <= 4:
            break
        else:
            pass

    output_nodes: list[SpecNode] = remove_duplicates(nodes)

    return output_nodes


def generate_value_row_vectors(
    num_vars: int, min_num_true: int, max_num_true: int
) -> list[list[bool]]:
    value_vectors: list[list[bool]] = []
    for num_true in [i for i in range(min_num_true, max_num_true + 1)]:
        for comb in combinations([i for i in range(0, num_vars)], num_true):
            zero_vector: list[bool] = [False for _ in range(num_vars)]

            for ind in comb:
                zero_vector[ind] = True

            value_vectors.append(zero_vector)

    return value_vectors


def create_neighbor_masks(
    num_vars: int, exclusions: list[Exclusion]
) -> tuple[ValueTable, ...]:
    default_row: list[bool] = [False for _ in range(num_vars)]

    mask_rows: list[list[bool]] = create_neighbor_mask_rows(num_vars)

    neg_masks: list[ValueTable] = (
        [ValueTable(mask_row, default_row, default_row) for mask_row in mask_rows]
        if "negation" not in exclusions
        else []
    )
    group_masks: list[ValueTable] = (
        [ValueTable(mask_row, default_row, default_row) for mask_row in mask_rows]
        if "group" not in exclusions
        else []
    )
    temporal_masks: list[ValueTable] = (
        [ValueTable(default_row, default_row, mask_row) for mask_row in mask_rows]
        if "temporal" not in exclusions
        else []
    )

    neighbor_masks: tuple[ValueTable, ...] = tuple(
        neg_masks + group_masks + temporal_masks
    )

    return neighbor_masks


def create_neighbor_mask_rows(num_vars: int) -> list[list[bool]]:
    neighbor_mask_rows: list[list[bool]] = []

    for i in range(num_vars):
        base: list[bool] = [False for _ in range(num_vars)]
        base[i] = True
        neighbor_mask_rows.append(base)

    return neighbor_mask_rows


def reverse_star(node: SpecNode) -> SpecNode:
    reversed_node = SpecNode(
        node.value_table, bool((node.star + 1) % 2), node.combination
    )
    return reversed_node


def reverse_stars(node: SpecNode) -> list[SpecNode]:
    reversed_nodes: list[SpecNode] = [
        SpecNode(
            node.value_table,
            not node.F_star,
            node.G_star,
            node.combination,
            node.exclusions,
            node.vars,
        ),
        SpecNode(
            node.value_table,
            node.F_star,
            not node.G_star,
            node.combination,
            node.exclusions,
            node.vars,
        ),
    ]

    return reversed_nodes


def other_combinations(node: SpecNode) -> list[SpecNode]:
    comb: Combination = node.combination
    possible_combs: list[Combination] = [0, 1, 2]
    combs: list[Combination] = list(set(possible_combs) - set([comb]))
    nodes: list[SpecNode] = [
        SpecNode(
            node.value_table, node.F_star, node.G_star, comb, node.exclusions, node.vars
        )
        for comb in combs
    ]

    return nodes


def find_neighbor_nodes_specs(
    node: SpecNode,
    neighbor_masks: tuple[ValueTable, ...],
    eliminate_nonsensical_specs: bool = False,
) -> tuple[list[SpecNode], list[str]]:
    nodes: list[SpecNode] = find_neighbors(node, neighbor_masks)
    specs: list[str] = nodes2specs(nodes)

    if eliminate_nonsensical_specs:
        nodes_tmp: list[SpecNode] = []
        specs_tmp: list[str] = []

        for node, spec in zip(nodes, specs):
            if not is_nonsensical_spec(spec, ["psi_ba_ra", "psi_ra_bf"]):
                nodes_tmp.append(node)
                specs_tmp.append(spec)
            else:
                pass

        nodes = nodes_tmp
        specs = specs_tmp

    else:
        pass

    return nodes, specs


def find_neighbors(
    node: SpecNode, neighbor_masks: tuple[ValueTable, ...]
) -> list[SpecNode]:
    neighbor_value_tables: list[ValueTable] = []

    for mask in neighbor_masks:
        temporal_row: list[bool] = (
            np.array(node.value_table.temporal) ^ np.array(mask.temporal)
        ).tolist()

        if len(list(set(temporal_row))) == 2:
            neg_row: list[bool] = (
                np.array(node.value_table.neg) ^ np.array(mask.neg)
            ).tolist()
            group_row: list[bool] = (
                np.array(node.value_table.group) ^ np.array(mask.group)
            ).tolist()

            neighbor_value_tables.append(ValueTable(neg_row, group_row, temporal_row))
        else:
            pass

    neighbor_nodes: list[SpecNode] = (
        [node]
        + [
            SpecNode(
                table,
                node.F_star,
                node.G_star,
                node.combination,
                node.exclusions,
                node.vars,
            )
            for table in neighbor_value_tables
        ]
        + reverse_stars(node)
        # + other_combinations(node)
    )

    output_nodes: list[SpecNode] = remove_duplicates(neighbor_nodes)

    return output_nodes


def remove_duplicates(nodes: list[SpecNode]) -> list[SpecNode]:
    sorting_dict: dict[str, SpecNode] = {}

    for node in nodes:
        sorting_dict[node2spec(node)] = node

    output_nodes: list[SpecNode] = list(sorting_dict.values())

    return output_nodes


def find_additional_neighbors(neighbor_nodes: list[SpecNode]) -> list[SpecNode]:
    masked_nodes: list[SpecNode] = neighbor_nodes[1:-1]

    additional_nodes: list[SpecNode] = []

    for node in masked_nodes:
        additional_nodes += reverse_stars(node)

    return additional_nodes


def table2spec(vars: tuple[str, ...], node: SpecNode):
    neg_row: list[int] = node.value_table[0, :].tolist()
    star_row: list[int] = node.value_table[1, :].tolist()

    negated_vars: list[str] = [
        ("!" if neg else "") + var for var, neg in zip(vars, neg_row)
    ]
    t_vars: list[str] = []
    f_vars: list[str] = []

    for var, star in zip(negated_vars, star_row):
        if star == 1:
            t_vars.append(var)
        elif star == 0:
            f_vars.append(var)
        else:
            raise Exception("star value should be either 0 or 1")

    group_connector: str = "&" if node.star else "|"
    t_group_star: bool = (
        not node.star
        if len(vars) <= 3
        else node.star if node.combination == 1 else not node.star
    )
    f_group_star: bool = (
        not node.star
        if len(vars) <= 3
        else node.star if node.combination == 2 else not node.star
    )
    t_connector: str = "&" if t_group_star else "|"
    f_connector: str = "&" if f_group_star else "|"
    t_spec: str = t_connector.join(t_vars)
    f_spec: str = f_connector.join(f_vars)

    spec: str = (
        "(" + t_spec + ")" + group_connector + "(" + f_spec + ")"
        if t_vars and f_vars
        else t_spec if t_vars else f_spec
    )

    return spec


def nodes2specs(nodes: list[SpecNode]) -> list[str]:
    specs: list[str] = [node2spec(node) for node in nodes]

    return specs


def node2spec(node: SpecNode) -> str:
    neg_row = node.value_table.neg
    group_row = node.value_table.group
    temporal_row = node.value_table.temporal
    vars = node.vars

    negated_vars: list[str] = (
        list(vars)
        if "negation" in node.exclusions
        else [("!" if neg else "") + var for var, neg in zip(vars, neg_row)]
    )

    F_vars: list[str] = []
    F_grouping: list[bool] = []
    G_vars: list[str] = []
    G_grouping: list[bool] = []

    for var, is_in_F, grouping in zip(negated_vars, temporal_row, group_row):
        if is_in_F:
            F_vars.append(var)
            F_grouping.append(grouping)
        else:
            G_vars.append(var)
            G_grouping.append(grouping)

    is_group_excluded: bool = "group" in node.exclusions

    f_group_spec: str = form_group(F_vars, F_grouping, node.F_star, is_group_excluded)
    g_group_spec: str = form_group(G_vars, G_grouping, node.G_star, is_group_excluded)

    if f_group_spec == "":
        print(node)
        raise Exception("F has no vars")
    elif g_group_spec == "":
        print(node)
        raise Exception("G has no vars")
    else:
        pass

    spec: str = "F(" + f_group_spec + ") & G(" + g_group_spec + ")"

    return spec


def spec2node(spec: str, dir_name: str) -> SpecNode:
    node_path: str = f"out/nodes/{dir_name}/{spec2title(spec)}.pkl"

    with open(node_path, "rb") as f:
        node: SpecNode = pickle.load(f)

    return node


def form_group(
    negated_vars: list[str], group_row: list[bool], group_star: bool, is_excluded: bool
) -> str:
    group_connector: str = "&" if group_star else "|"
    group_spec: str = ""

    if is_excluded:
        group_spec = group_connector.join(negated_vars)
    else:
        t_vars: list[str] = []
        f_vars: list[str] = []

        for var, group in zip(negated_vars, group_row):
            if group == True:
                t_vars.append(var)
            else:
                f_vars.append(var)

        var_connector: str = "|" if group_star else "&"
        t_spec: str = var_connector.join(t_vars)
        f_spec: str = var_connector.join(f_vars)
        group_spec = (
            "(" + t_spec + ")" + group_connector + "(" + f_spec + ")"
            if t_vars and f_vars
            else t_spec if t_vars else f_spec
        )

    return group_spec


def compare_nodes(input: SpecNode, targets: list[SpecNode]) -> bool:
    is_included: bool = False

    for target in targets:
        if input.star == target.star:
            if np.array_equal(input.value_table, target.value_table):
                is_included = True
            else:
                pass
        else:
            pass

    return is_included


def eliminate_searched_specs(
    all_specs: list[str], file_dir: str, file_prefix: str, file_ext: str
) -> tuple[list[str], list[int]]:
    unsearched_specs: list[str] = []
    unsearchd_inds: list[int] = []

    for i, spec in enumerate(all_specs):
        if not os.path.isfile(file_dir + file_prefix + "_{}.".format(i) + file_ext):
            unsearched_specs.append(spec)
            unsearchd_inds.append(i)
        else:
            pass

    return unsearched_specs, unsearchd_inds

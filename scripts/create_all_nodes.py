import os
import pickle

import itertools

from tl_search.common.io import spec2title
from tl_search.common.typing import Combination, Exclusion, SpecNode, ValueTable
from tl_search.search.neighbor import node2spec

predicates: tuple[str, ...] = (
    "psi_gl",
    "psi_hz",
    "psi_vs",
)
exclusions: list[Exclusion] = ["group"]

# Create all possible nodes
num_predicates: int = len(predicates)

neg_rows: list[list[bool]] = (
    list(itertools.product([False, True], repeat=num_predicates))
    if "negation" not in exclusions
    else [[False] * num_predicates]
)
group_rows: list[list[bool]] = (
    list(itertools.product([False, True], repeat=num_predicates))
    if "group" not in exclusions
    else [[False] * num_predicates]
)
temporal_rows: list[list[bool]] = list(
    itertools.product([False, True], repeat=num_predicates)
)
# Remove temporal rows that are all False or True
temporal_rows = [row for row in temporal_rows if any(row) and not all(row)]

f_stars: list[bool] = [False, True]
g_stars: list[bool] = [False, True]

for neg_row in neg_rows:
    for group_row in group_rows:
        for temporal_row in temporal_rows:
            value_table = ValueTable(neg_row, group_row, temporal_row)
            for f_star in f_stars:
                for g_star in g_stars:
                    node = SpecNode(
                        value_table=value_table,
                        F_star=f_star,
                        G_star=g_star,
                        combination=0,
                        exclusions=exclusions,
                        vars=predicates,
                    )

                    print(node2spec(node))

                    node_path: str = f"out/nodes/goal/{spec2title(node2spec(node))}.pkl"

                    os.makedirs(os.path.dirname(node_path), exist_ok=True)
                    with open(node_path, "wb") as f:
                        pickle.dump(node, f)

                    print(spec2title(node2spec(node)))

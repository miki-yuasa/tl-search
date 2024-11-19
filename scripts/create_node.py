import os
import pickle
from tl_search.common.io import spec2title
from tl_search.common.typing import Combination, Exclusion, SpecNode, ValueTable
from tl_search.search.neighbor import node2spec

# create node for "F((psi_ba_ra|!psi_ba_rf)&(!psi_ba_bt)) & G(psi_ra_bf)"
predicates: tuple[str, ...] = (
    "psi_ba_ra",
    "psi_ba_bt",
    "psi_ba_rf",
    "psi_ra_bf",
)
neg_row: list[bool] = [False, False, False, True]
group_row: list[bool] = [False, False, True, True]
temporal_row: list[bool] = [False, False, True, True]

value_table = ValueTable(neg_row, group_row, temporal_row)

node = SpecNode(value_table, False, True, 0, [], predicates)

print(node2spec(node))

enemy_policy_mode: str = "patrol"

node_path: str = f"out/nodes/{enemy_policy_mode}/{spec2title(node2spec(node))}.pkl"

os.makedirs(os.path.dirname(node_path), exist_ok=True)
with open(node_path, "wb") as f:
    pickle.dump(node, f)

print(spec2title(node2spec(node)))

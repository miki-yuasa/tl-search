from tl_search.common.typing import Exclusion
from tl_search.search.neighbor import create_all_nodes, node2spec


predicates: tuple[str, ...] = ("psi_ego_goal", "psi_ego_adv", "psi_ego_wall")
exclusions: list[Exclusion] = ["group"]

all_nodes = create_all_nodes(predicates, exclusions)

all_specs = [node2spec(node) for node in all_nodes]

unique_specs = list(set(all_specs))

print(len(all_specs))

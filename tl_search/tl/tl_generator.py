import itertools
import copy

from tl_search.tl.tl_parser import is_var, tokenize


def generate_tl_specs(phi_task: str, atom_prop_dict: dict[str, str]) -> list[str]:
    atom_props_all: list[str] = list(atom_prop_dict.keys())
    task_tokens: list[str] = tokenize(phi_task)
    task_props: list[str] = [
        token for token in task_tokens if is_var(token, atom_prop_dict)
    ]

    atom_props: list[str] = [prop for prop in atom_props_all if prop not in task_props]
    A: list[int] = list(range(len(atom_props)))
    A_len: int = len(A)
    tl_specs: list[str] = format_tl_specs(
        phi_task, generate_global_specs(("", ""), atom_props)
    )

    for i in A:
        for j in range(i + 1, A_len):
            global_specs: list[str] = generate_global_specs(
                (atom_props[i], atom_props[j]), atom_props
            )
            tl_specs += format_tl_specs(phi_task, global_specs)

    return tl_specs


def generate_global_specs(
    target_props: tuple[str, str], atom_props: list[str]
) -> list[str]:
    all_props = copy.copy(atom_props)
    if target_props == ("", ""):
        pass
    else:
        all_props.remove(target_props[0])
        all_props.remove(target_props[1])
    remained_props: list[str] = all_props
    remained_specs: list[str] = [" & ".join(all_props)]
    for i in range(1, len(all_props) + 1):
        combs: list[tuple[str, ...]] = list(itertools.combinations(remained_props, i))
        for comb in combs:
            remained_props_tmp = copy.copy(remained_props)
            for prop in comb:
                remained_props_tmp.remove(prop)
                remained_props_tmp.append("!" + prop)
            remained_specs.append(" & ".join(remained_props_tmp))

    inverse_target_props: list[str] = list(target_props)
    inverse_target_props.reverse()
    target_specs = [
        "(" + "->".join(target_props) + ")",
        "!(" + "->".join(target_props) + ")",
        "(" + "->".join(inverse_target_props) + ")",
        "!(" + "->".join(inverse_target_props) + ")",
    ]

    global_specs: list[str] = []
    if target_props == ("", ""):
        global_specs += remained_specs
    else:
        for target_spec in target_specs:
            global_specs += [
                target_spec + " & " + remained_spec for remained_spec in remained_specs
            ]

    return global_specs


def format_tl_specs(phi_task: str, global_specs: list[str]) -> list[str]:
    tl_specs = [
        "F " + phi_task + " & G( " + global_spec + ")" for global_spec in global_specs
    ]
    return tl_specs

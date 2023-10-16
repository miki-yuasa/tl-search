import re, sys
from collections import deque
from typing import Callable, Union, cast, Pattern
from tl_search.envs.typing import FieldObj

from tl_search.tl.synthesis import TLAutomaton
from tl_search.common.typing import SymbolProp, ObsProp, MapLocations


class Parsers:
    def __init__(self):
        self.splitter: Pattern = re.compile(r"[\s]*(\d+|\w+|.)")

        self.parentheses: list[str] = ["(", ")"]

        self.symbols: dict[str, SymbolProp] = {
            "!": SymbolProp(3, lambda x: -x),
            "|": SymbolProp(1, lambda x, y: max(x, y)),
            "&": SymbolProp(2, lambda x, y: min(x, y)),
            "<": SymbolProp(2, lambda x, y: y - x),
            ">": SymbolProp(2, lambda x, y: x - y),
            "->": SymbolProp(2, lambda x, y: max(-x, y)),
            "<-": SymbolProp(2, lambda x, y: max(x, -y)),
        }


__parsers = Parsers()


def atom_tl_ob2rob(
    aut: TLAutomaton, locations: MapLocations | FieldObj
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute robustnesses (rho) of the atomic TL porlocitions (psi) based on
    observation.

    Parameters:
    aut (TLAutomaton): automaton from a TL spec
    locations (MapLocations): object locations in a map

    Returns:
    atom_rob_dict (dict[str, float]): dictionary of robustnesses of atomic propositions
    obs_dict (dict[str, float]): dictionary of observation and its values
    """

    location_dict: dict[str, float] = locations._asdict()

    obs_dict: dict[str, float] = {
        obs.name: obs.func(*[location_dict[arg] for arg in obs.args])
        for obs in aut.obs_props
    }

    atom_rob_dict: dict[str, float] = {
        atom_props_key: tl2rob(aut.atom_prop_dict[atom_props_key], obs_dict)
        for atom_props_key in aut.atom_prop_dict.keys()
    }

    return (atom_rob_dict, obs_dict)


def get_used_tl_props(
    tl_spec: str, atom_prop_dict_all: dict[str, str], obs_props_all: list[ObsProp]
) -> tuple[list[ObsProp], dict[str, str]]:
    prop_vars = get_vars(tokenize(tl_spec), atom_prop_dict_all)
    used_obs = get_used_obs(prop_vars, atom_prop_dict_all, obs_props_all)

    atom_prop_dict: dict[str, str] = {
        prop_var: atom_prop_dict_all[prop_var] for prop_var in prop_vars
    }
    obs_props: list[ObsProp] = [
        obs_prop for obs_prop in obs_props_all if obs_prop.name in used_obs
    ]

    return obs_props, atom_prop_dict


def tl2rob(spec: str, var_dict: Union[dict[str, str], dict[str, float]]) -> float:
    """
    Parsing a TL spec to its robustness

    Parameters:


    Returns:

    """

    token_list: list[str] = tokenize(spec)
    parsed_tokens: list[str] = parse(token_list, var_dict)
    result: float = evaluate(parsed_tokens, var_dict)

    return result


# Check if a token is a parenthesis
def is_parentheses(
    s: str, PARENTHESES: list[str] = __parsers.parentheses, **kwargs
) -> bool:
    if "index" in kwargs:
        return s is PARENTHESES[kwargs["index"]]
    return s in PARENTHESES


# Check if a token is symbol
def is_symbol(s: str, SYMBOLS: dict[str, SymbolProp] = __parsers.symbols) -> bool:
    return s in SYMBOLS


# Check if a token is number
def is_num(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


# Check if a token is variable
def is_var(s: str, var_dict: Union[dict[str, str], dict[str, float]]) -> bool:
    return s in var_dict.keys()


# Priorities of Symbol
def get_priority(s: str, SYMBOLS: dict[str, SymbolProp] = __parsers.symbols) -> int:
    return SYMBOLS[s].priority


# Funcion of Symbol
def get_func(s: str, SYMBOLS: dict[str, SymbolProp] = __parsers.symbols) -> Callable:
    return SYMBOLS[s].func


# Get only vars from a token list
def get_vars(
    token_list: list[str], var_dict: Union[dict[str, str], dict[str, float]]
) -> list[str]:
    vars: list[str] = [token for token in token_list if is_var(token, var_dict)]
    vars_out: list[str] = list(set(vars))
    return vars_out


# Get used observations of given atomic props
def get_used_obs(
    prop_list: list[str], prop_dict: dict[str, str], obs_props: list[ObsProp]
) -> list[str]:
    obs_vars_all: list[str] = [obs_prop.name for obs_prop in obs_props]
    obs_set: set[str] = set()

    for prop in prop_list:
        spec = prop_dict[prop]
        tokens = tokenize(spec)
        obs_vars: list[str] = [token for token in tokens if token in obs_vars_all]
        obs_set |= set(obs_vars)

    return list(obs_set)


# Tokenize the spec
def tokenize(spec: str, SPLITTER: Pattern = __parsers.splitter) -> list[str]:
    token_list_tmp = deque(SPLITTER.findall(spec))
    token_list: list[str] = []
    while token_list_tmp:
        token = token_list_tmp.popleft()
        if token == ".":
            if is_num(token_list[-1]) and is_num(token_list_tmp[0]):
                token_list.append(token_list.pop() + token + token_list_tmp.popleft())
            else:
                print(
                    "Error: invalid '.' in the spec",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif token == "-":
            if token_list[-1] == "<":
                token_list.append(token_list.pop + token)
            elif token_list_tmp[0] == ">":
                token_list.append(token + token_list_tmp.popleft())
            else:
                print(
                    "Error: invalid '-' in the spec",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            token_list.append(token)
    return token_list


def parse(
    token_list: list[str],
    var_dict: Union[dict[str, str], dict[str, float]],
    PARENTHESES: list[str] = __parsers.parentheses,
    SYMBOLS: dict[str, SymbolProp] = __parsers.symbols,
) -> list[str]:
    """
    Convert the token to Reverse Polish Notation
    """
    stack: list[str] = []
    output_stack: list[str] = []

    for i in range(len(token_list)):
        token: str = token_list.pop(0)
        # If a number, push to the output stack
        if is_num(token) | is_var(token, var_dict):
            output_stack.append(token)
        # If a starting parenthesis, push to stack
        elif is_parentheses(token, PARENTHESES, index=0):
            stack.append(token)
        # If an ending parenthesis, pop and add to the
        # output stack until the starting parenthesis
        # comes.
        elif is_parentheses(token, PARENTHESES, index=1):
            for i in range(len(stack)):
                symbol: str = stack.pop()
                if is_parentheses(symbol, PARENTHESES, index=0):
                    break
                output_stack.append(symbol)
        # If the read token's priority is less than that
        # of the one in the end of the stack, pop from
        # the stack, add to the output stack, and then
        # add to the stack
        elif (
            stack
            and is_symbol(stack[-1], SYMBOLS)
            and get_priority(token, SYMBOLS) <= get_priority(stack[-1], SYMBOLS)
        ):
            symbol = stack.pop()
            output_stack.append(symbol)
            stack.append(token)
        # Push the others to the stack
        else:
            stack.append(token)
    # Finally, add the stack to the ouput stack
    while stack:
        output_stack.append(stack.pop(-1))
    return output_stack


def evaluate(
    parsed_tokens: list[str],
    var_dict: Union[dict[str, str], dict[str, float]],
    SYMBOLS: dict[str, SymbolProp] = __parsers.symbols,
) -> float:
    """
    Checking from the start, get tokens from the stack and
    compute there if there is a symbol
    """
    output: list[str | float] = [token for token in parsed_tokens]
    cnt: int = 0
    while len(output) != 1:
        if is_symbol(output[cnt], SYMBOLS):
            symbol: str = output.pop(cnt)
            num_args: int = SYMBOLS[symbol].func.__code__.co_argcount
            target_index: int = cnt - num_args
            args: list[float] = []
            for i in range(num_args):
                arg: str = output.pop(target_index)
                if is_num(arg):
                    args.append(float(arg))
                elif arg.isascii():
                    args.append(cast(float, var_dict[arg]))
                else:
                    print("Error: the type of the token should be either float or str.")
                    print(type(arg), file=sys.stderr)
                    sys.exit(1)
            result = get_func(symbol, SYMBOLS)(*args)
            output.insert(target_index, str(result))
            cnt = target_index + 1
        else:
            cnt += 1

    final_result: float = (
        float(var_dict[output[0]])
        if isinstance(output[0], str) and output[0] in var_dict.keys()
        else float(output[0])
    )

    return final_result


# spec = "psi_ba_ra&!psi_ba_ob&!psi_ba_rt&!psi_ba_wa&!psi_ra_bt | psi_ba_ra&!psi_ba_ob&!psi_ba_rt&!psi_ba_wa&!psi_ra_bf"
# rob_dict = {
#    "psi_ba_ra": -5.0710678118654755,
#    "psi_ba_rf": -6.5710678118654755,
#    "psi_ba_rt": -3.5,
#    "psi_ra_bf": -8.055385138137417,
#    "psi_ba_ob": -1.5,
#    "psi_ba_wa": -2.5,
#    "psi_ra_bt": -3,
# }
# tl2rob(spec, rob_dict)
# token_list: list[str] = tokenize(spec, __parsers.splitter)
# var_list = get_vars(token_list, rob_dict)
# print(var_list)

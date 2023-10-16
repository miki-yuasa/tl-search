from typing import Union
from tl_search.common.typing import Hyperparameters


def replace_hyperparam(
    target: str,
    value: Union[float, int, bool, tuple[int, int]],
    default_value_table: dict[
        str,
        Union[
            float,
            int,
            bool,
            tuple[int, int],
            list[float],
            list[int],
            list[tuple[int, int]],
        ],
    ],
) -> Hyperparameters:
    default_value_table[target] = value

    return Hyperparameters(
        default_value_table["learning_rate"],
        default_value_table["n_step_batch_pair"][0],
        default_value_table["n_step_batch_pair"][1],
        default_value_table["n_epochs"],
        default_value_table["gamma"],
        default_value_table["gae_lambda"],
        default_value_table["clip_range"],
        default_value_table["ent_coef"],
        default_value_table["vf_coef"],
        default_value_table["max_grad_norm"],
    )

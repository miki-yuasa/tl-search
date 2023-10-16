from typing import Union

from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np

from policies.known_policy import KnownPolicy
from policies.unknown_policy import UnknownPolicy
from tl_search.evaluation.filter import filter_rewards
from train.train import spec_train, train_spec
from tl_search.common.logger import log_experiment_data
from tl_search.common.typing import (
    EnemyPolicyMode,
    EnvProps,
    ExperimentLogger,
    ModelProps,
    ObsProp,
    ActionProbsSpec,
    PositiveRewardLogger,
    RLAlgorithm,
    SaveMode,
)


def train_evaluate_multiprocess(
    num_process: int,
    description: str,
    tl_specs: list[str],
    tl_specs_done: list[str],
    kl_divs_all: list[list[float]],
    tl_action_probs_all: list[ActionProbsSpec],
    ori_action_probs_all: list[ActionProbsSpec],
    model_names_all: list[list[str]],
    mean_rewards_all: list[list[float]],
    std_rewards_all: list[list[float]],
    experiment: int,
    gpus: tuple[int, ...],
    target: KnownPolicy | UnknownPolicy | ModelProps,
    enemy_policy: EnemyPolicyMode,
    num_sampling: int,
    num_replicates: int,
    total_timesteps: int,
    atom_prop_dict_all: dict[str, str],
    obs_props_all: list[ObsProp],
    log_dir: str,
    model_savename: str,
    model_save_mode: SaveMode,
    reward_curve_plot_mode: SaveMode,
    reward_curve_plot_savename: str,
    animation_mode: SaveMode,
    animation_savename: str,
    is_gui_shown: bool,
    is_trap_state_filtered: bool,
    is_reward_calculated: bool,
    is_initialized_in_both_territories: bool,
    replicate_seeds: list[int],
    is_similarity_compared: bool = True,
    is_from_all_territories: bool = False,
    spec_ind_start: int = 0,
    learning_rate: float = 1e-5,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: Union[float, None] = None,
    ent_coef: float = 0.0,
    vf_coef: float = 1.0,
    max_grad_norm: float = 0.75,
) -> tuple[ExperimentLogger, PositiveRewardLogger]:
    all_specs: list[str] = tl_specs_done + tl_specs
    inputs = [
        (
            spec_ind + spec_ind_start,
            gpus[int(spec_ind // (len(tl_specs) / len(gpus)))],
            target,
            enemy_policy,
            num_sampling,
            num_replicates,
            total_timesteps,
            tl_spec,
            atom_prop_dict_all,
            obs_props_all,
            log_dir,
            model_savename,
            model_save_mode,
            reward_curve_plot_mode,
            reward_curve_plot_savename,
            is_trap_state_filtered,
            is_reward_calculated,
            is_initialized_in_both_territories,
            animation_mode,
            animation_savename,
            is_gui_shown,
            replicate_seeds,
            is_similarity_compared,
            is_from_all_territories,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            ent_coef,
            vf_coef,
            max_grad_norm,
        )
        for spec_ind, tl_spec in enumerate(tl_specs)
    ]

    with Pool(num_process) as p:
        results = p.map(spec_train_evaluate_wrapper, inputs)

    for result in results:
        (
            kl_divs_spec,
            tl_action_probs_spec,
            ori_action_probs_spec,
            mean_rewards_spec,
            std_rewards_spec,
            model_names_spec,
        ) = result

        kl_divs_all.append(kl_divs_spec)
        tl_action_probs_all.append(tl_action_probs_spec)
        ori_action_probs_all.append(ori_action_probs_spec)
        mean_rewards_all.append(mean_rewards_spec)
        std_rewards_all.append(std_rewards_spec)
        model_names_all.append(model_names_spec)

    experiment_data = log_experiment_data(
        type(target).__name__,
        experiment,
        description,
        all_specs,
        kl_divs_all,
        tl_action_probs_all,
        ori_action_probs_all,
        model_names_all,
        mean_rewards_all,
        std_rewards_all,
    )

    positive_reward_data = filter_rewards(
        0,
        all_specs,
        np.mean(np.array(kl_divs_all), axis=1).astype(object),
        mean_rewards_all,
        []
        if model_save_mode == "disabled"
        else [models[0] for models in model_names_all],
    )

    return experiment_data, positive_reward_data


def spec_train_evaluate_wrapper(
    inputs: tuple[
        int,
        int,
        KnownPolicy | UnknownPolicy | ModelProps,
        EnemyPolicyMode,
        int,
        int,
        int,
        str,
        dict[str, str],
        list[ObsProp],
        str,
        str,
        SaveMode,
        SaveMode,
        str,
        Union[bool, None],
        bool,
        bool,
        SaveMode,
        str,
        bool,
        list[int],
        bool,
        bool,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        Union[float, None],
        float,
        float,
        float,
    ]
) -> tuple[
    list[float], ActionProbsSpec, ActionProbsSpec, list[float], list[float], list[str]
]:
    (
        spec_ind,
        gpu,
        target,
        enemy_policy,
        num_sampling,
        num_replicates,
        total_timesteps,
        tl_spec,
        atom_prop_dict_all,
        obs_props_all,
        log_dir,
        model_savename,
        model_save_mode,
        reward_curve_plot_mode,
        reward_curve_plot_savename,
        is_trap_state_filtered,
        is_reward_calculated,
        is_initialized_in_both_territories,
        animation_mode,
        animation_savename,
        is_gui_shown,
        replicate_seeds,
        is_similarity_compared,
        is_from_all_territories,
        learning_rate,
        n_steps,
        batch_size,
        n_epochs,
        gamma,
        gae_lambda,
        clip_range,
        clip_range_vf,
        ent_coef,
        vf_coef,
        max_grad_norm,
    ) = inputs

    return spec_train(
        spec_ind,
        gpu,
        target,
        enemy_policy,
        num_sampling,
        num_replicates,
        total_timesteps,
        tl_spec,
        atom_prop_dict_all,
        obs_props_all,
        log_dir,
        model_savename,
        model_save_mode,
        reward_curve_plot_mode,
        reward_curve_plot_savename,
        is_trap_state_filtered,
        is_reward_calculated,
        is_initialized_in_both_territories,
        animation_mode,
        animation_savename,
        is_gui_shown,
        replicate_seeds,
        is_similarity_compared,
        is_from_all_territories,
        learning_rate,
        n_steps,
        batch_size,
        n_epochs,
        gamma,
        gae_lambda,
        clip_range,
        clip_range_vf,
        ent_coef,
        vf_coef,
        max_grad_norm,
    )


def train_multipocesss(
    num_process: int,
    rl_algorithm: RLAlgorithm,
    env_props_list: list[EnvProps],
    spec_inds: list[int],
    gpu: int,
    num_replicates: int,
    total_timesteps: int,
    log_dir: str,
    model_savename: str,
    reward_curve_plot_mode: SaveMode,
    reward_curve_plot_savename: str,
    is_initialized_in_both_territories: bool,
    animation_mode: SaveMode,
    animation_savename: str,
    is_gui_shown: bool,
    replicate_seeds: list[int],
    learning_rate: float = 1e-5,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: float | None = None,
    ent_coef: float = 0.0,
    vf_coef: float = 1.0,
    max_grad_norm: float = 0.75,
) -> list[list[ModelProps]]:
    inputs = [
        (
            env_props,
            rl_algorithm,
            spec_ind,
            gpu,
            num_replicates,
            total_timesteps,
            log_dir,
            model_savename,
            reward_curve_plot_mode,
            reward_curve_plot_savename,
            is_initialized_in_both_territories,
            animation_mode,
            animation_savename,
            is_gui_shown,
            replicate_seeds,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            ent_coef,
            vf_coef,
            max_grad_norm,
        )
        for spec_ind, env_props in zip(spec_inds, env_props_list)
    ]

    with Pool(num_process) as p:
        results: list[list[ModelProps]] = p.map(train_spec_wrapper, inputs)

    return results


def train_spec_wrapper(
    input: tuple[
        EnvProps,
        RLAlgorithm,
        int,
        int,
        int,
        int,
        str,
        str,
        SaveMode,
        str,
        bool,
        SaveMode,
        str,
        bool,
        list[int],
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        float | None,
        float,
        float,
        float,
    ]
) -> list[ModelProps]:
    return train_spec(*input)

from typing import Any

from stable_baselines3 import SAC

from tl_search.envs.tl_parking import TLAdversarialParkingEnv


def sample_obs(
    env: TLAdversarialParkingEnv, model: SAC, num_samples: int
) -> list[dict[str, Any]]:
    obs_list: list[dict[str, Any]] = []

    sample_counter: int = 0

    while sample_counter < num_samples:
        obs, _ = env.reset()
        while True:
            action = model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, _ = env.step(action)

            if (
                obs["aut_state"] not in env.aut.goal_states
                or obs["aut_state"] not in env.aut.trap_states
            ):
                obs_list.append(obs)
                sample_counter += 1
            else:
                pass

            if terminated or truncated:
                break

    assert len(obs_list) == num_samples

    return obs_list

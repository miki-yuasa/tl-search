from highway_env.envs.common.action import Action

from tl_search.common.typing import AutomatonStateStatus, ObsProp
from tl_search.envs.parking import AdversarialParkingEnv
from tl_search.tl.synthesis import TLAutomaton


class TLAdversarialParkingEnv(AdversarialParkingEnv):
    def __init__(
        self,
        tl_spec: str,
        obs_props: list[ObsProp],
        atom_pred_dict: dict[str, str],
        config: dict | None = None,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__(config, render_mode)

        self._tl_spec: str = tl_spec
        self._obs_props: list[ObsProp] = obs_props
        self._atom_pred_dict: dict[str, str] = atom_pred_dict

        self.aut = TLAutomaton(tl_spec, atom_pred_dict, obs_props)

    def reset(self) -> tuple[dict[str, float], dict[str, float]]:
        obs, info = super().reset()

        self._aut_state: int = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._status: AutomatonStateStatus = "intermediate"

        return obs, info

    def step(self, action: Action):
        obs, reward, terminated, truncated, info = super().step(action)

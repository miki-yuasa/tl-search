import numpy as np
from numpy.typing import NDArray

from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1

from tl_search.common.typing import ObsProp, AutomatonStateStatus
from tl_search.tl.reward import tl_reward
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.tl_parser import tl2rob


class TLGoalLevel1(GoalLevel1):
    def __init__(self, config) -> None:
        self.tl_spec: str = ""
        super().__init__(config)

        obs_props: list[ObsProp] = [
            ObsProp("d_gl", ["d_gl"], lambda d_goal: d_goal),
            ObsProp("d_hz", ["d_hz"], lambda d_haz: d_haz),
            ObsProp("d_vs", ["d_vs"], lambda d_vase: d_vase),
        ]

        atom_pred_dict: dict[str, str] = {
            "psi_gl": f"d_gl < 0.3",
            "psi_hz": f"d_hz < 0.2",
            "psi_vs": f"d_vs < 0.1",
        }

        self.aut = TLAutomaton(self.tl_spec, atom_pred_dict, obs_props)

    def specific_reset(self):
        self._aut_state = self.aut.start
        self._aut_state_traj: list[int] = [self._aut_state]
        self._aut_status: AutomatonStateStatus = "intermediate"

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        d_goal = self.dist_goal()
        d_haz_all: list[float] = [
            self.agent.dist_xy(pos) for pos in self._geoms["hazards"].pos
        ]
        d_haz: float = min(d_haz_all)
        d_vase_all: list[float] = [
            self.agent.dist_xy(pos) for pos in self._free_geoms["vases"].pos
        ]
        d_vase: float = min(d_vase_all)

        kin_dict: dict[str, float] = {
            "d_gl": d_goal,
            "d_hz": d_haz,
            "d_vs": d_vase,
        }

        atom_rob_dict, obs_dict = atom_tl_ob2rob(self.aut, kin_dict)
        reward, next_aut_state = tl_reward(
            atom_rob_dict, self.aut, self._aut_state, True
        )

        self._aut_state = next_aut_state
        self._aut_state_traj.append(next_aut_state)
        if next_aut_state in self.aut.goal_states:
            self._status = "goal"
        elif next_aut_state in self.aut.trap_states:
            self._status = "trap"
        else:
            self._status = "intermediate"

        return reward

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size or self._status in ["goal", "trap"]


def atom_tl_ob2rob(
    aut: TLAutomaton, kin_dict: dict[str, NDArray[np.float64] | np.float64]
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute robustnesses (rho) of the atomic TL porlocitions (psi) based on
    observation.

    Parameters
    ----------
    aut: TLAutomaton
        automaton from a TL spec
    kin_dict: dict[str, NDArray[np.float64]]
        Kinematic information of the agents

    Returns
    -------
    atom_rob_dict (dict[str, float]): dictionary of robustnesses of atomic propositions
    obs_dict (dict[str, float]): dictionary of observation and its values
    """

    obs_dict: dict[str, float] = {
        obs.name: obs.func(*[kin_dict[arg] for arg in obs.args])
        for obs in aut.obs_props
    }

    atom_rob_dict: dict[str, float] = {
        atom_props_key: tl2rob(aut.atom_prop_dict[atom_props_key], obs_dict)
        for atom_props_key in aut.atom_prop_dict.keys()
    }

    return (atom_rob_dict, obs_dict)

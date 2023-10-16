import unittest

import numpy as np

# from tl_search.tl.synthesis import TLAutomaton
# from tl_search.tl.constants import obs_props_all
# from tl_search.common.utils import kl_div


# class TestTLAutomaton(unittest.TestCase):
#     def setUp(self) -> None:
#         fight_range: float = 2.0
#         defense_range: float = 1.0
#         self.atom_prop_dict_all: dict[str, str] = {
#             "psi_ba_ra": "d_ba_ra < {}".format(fight_range),
#             # "psi_ba_rf": "d_ba_rf < 0.5",
#             "psi_ba_rt": "d_ba_rt < 0.5",
#             "psi_ra_bf": "d_ra_bf < {}".format(defense_range),
#             "psi_ra_bt": "d_ra_bt < 0.5",
#             "psi_ba_ob": "d_ba_ob < 0.5",
#             "psi_ba_wa": "d_ba_wa < 0.5",
#             # "psi_ba_obwa": "(d_ba_wa < 0.5)|(d_ba_ob < 0.5)",
#         }
#         self.obs_props = obs_props_all

#     def test_aut_init(self):
#         spec = "F(psi_ba_ra & !psi_ba_rt)& G(!psi_ba_ra&!psi_ba_rt&!psi_ra_bf&!psi_ra_bt&!psi_ba_ob&!psi_ba_wa)"
#         TLAutomaton(spec, self.atom_prop_dict_all, self.obs_props)

#     def test_kl_div(self):
#         val = kl_div(np.array([0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]))
#         print(val)


# if __name__ == "__main__":
#     unittest.main(verbosity=2)

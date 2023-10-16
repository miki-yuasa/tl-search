import unittest

from tl_search.search.filter import is_nonsensical_spec


class TestSearchFilter(unittest.TestCase):
    def test_is_nonsensical_spec(self):
        specs: list[str] = [
            "F(psi_ba_ra) & G(psi_ra_bf)",
            "F(psi_ra_bf) & G(psi_ba_ra & psi_ba_bf & psi_ba_bt)",
            "F(psi_ra_bf) & G((psi_ba_ra) | psi_ba_bf | psi_ba_bt)",
            "F((psi_ba_rf)|(!psi_ra_bf)) & G((psi_ba_ra&psi_ba_bf)|(!psi_ba_bt))",
            "F((!psi_ba_bt|!psi_ra_bf)&(!psi_ba_rf|psi_ba_bf)) & G(psi_ba_ra)",
            "F(psi_ba_ra&!psi_ba_bt) & G((!psi_ba_rf&!psi_ra_bf)|(psi_ba_bf))",
            "F((!psi_ba_bt)&(psi_ba_rf|!psi_ba_bf|psi_ra_bf)) & G(!psi_ba_ra)",
            "F((!psi_ba_bt)&(!psi_ba_ra|psi_ba_rf)) & G((psi_ba_bf)|(psi_ra_bf))",
        ]

        answers: list[bool] = [False, True, False, False, False, False, False, False]

        for spec, answer in zip(specs, answers):
            self.assertEqual(
                is_nonsensical_spec(spec, ["psi_ba_ra"]),
                answer,
            )

from tl_search.common.io import spec2title


spec: str = "F((!psi_ba_bt)&(psi_ba_rf)) & G((!psi_ba_ra)|(!psi_ra_bf))"

print(spec2title(spec))

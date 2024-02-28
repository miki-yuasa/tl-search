import numpy as np
from numpy.typing import NDArray

from tl_search.search.search_parking import gaussian_kl_div


target_action_path: str = (
    "out/data/search/dataset/action_probs_sac_parking_exp1_extended_full.npz"
)
rw_std: float = 10

# Load the action probabilities
action_means: np.ndarray = np.load(target_action_path)["gaus_means"]
action_stds: np.ndarray = np.load(target_action_path)["gaus_stds"]

rw_means: NDArray = action_means
rw_stds: NDArray = 5 * np.ones_like(action_stds)

# Calculate the KL divergence
kl_div: np.float_ = np.mean(
    gaussian_kl_div(rw_means, rw_stds, action_means, action_stds)
)
print(f"KL divergence: {kl_div}")

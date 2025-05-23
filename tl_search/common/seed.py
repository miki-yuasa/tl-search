import os

import random
import numpy as np
import torch


def torch_fix_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

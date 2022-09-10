from neurorobotics.utils import cpg_utils
from neurorobotics.utils import cv_utils
from neurorobotics.utils import env_utils
from neurorobotics.utils import point_cloud
from neurorobotics.utils import torch_utils
from neurorobotics.utils import visualise
from neurorobotics.utils import feature_extractors
from neurorobotics.utils import schedules
import torch
import numpy as np
import random

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG

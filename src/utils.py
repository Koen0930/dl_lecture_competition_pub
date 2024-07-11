import random
import numpy as np
import torch
import math

from torch import nn
import torch.nn.functional as F

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


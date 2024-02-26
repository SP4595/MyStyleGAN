import torch
import torch.nn as nn
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import random

def set_seed(seed_id : int) -> None:
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    np.random.seed(seed_id)
    random.seed(seed_id)
    torch.backends.cudnn.deterministic = True
import torch
import numpy as np
import random
import os
import glob


def fix_seed(seed:2333):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_files(folder:os.PathLike, pattern:str):
    """
        return all files with given patterns.
    """
    assert os.path.exists(folder), f"{folder} do not exists"
    return glob.glob(os.path.join(folder, pattern)).sort()
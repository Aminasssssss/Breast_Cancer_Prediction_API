import random
import numpy as np
import os

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_reproducibility_info():
    return {
        "random_seed": 42,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "numpy_version": np.__version__
    }

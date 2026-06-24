# ==============================================================================
# Seeding Module for Reproducible Experiments
# ==============================================================================

import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python's built-in 'random' library, NumPy, and 
    TensorFlow to ensure reproducibility across experimental runs.

    Also sets key environment variables to encourage deterministic execution
    within TensorFlow/CUDNN operations.

    Args:
        seed (int): The seed value to be set. Defaults to 42.
    """
    # 1. Set standard Python seed
    random.seed(seed)
    
    # 2. Set environment variable for hash-based operations reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 3. Set NumPy random seed
    np.random.seed(seed)
    
    # 4. Set TensorFlow global seed
    tf.random.set_seed(seed)
    
    # 5. Configure environment variables for TF/CUDNN deterministic operations
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    
    print(f"[SEEDING] Global seed set to: {seed} (python-random, numpy, tensorflow).")

import random
import os
import numpy as np
import torch


"""
Utility functions for the project.
"""

"""
Seed everything for reproducibility.
"""
def seed_everything(seed=42):
    # 1. Base Python environment seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch CPU and GPU seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe for multi-GPU setups
    
    # 3. Configure CUDA reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. Force strict deterministic operations within PyTorch (Throws warnings/errors if non-reproducible functions are used)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"✅ Environment successfully seeded with seed: {seed}")
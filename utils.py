

import jax
import numpy as onp
import jax.numpy as jnp
import random
import torch
import torch.cuda
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return jax.random.PRNGKey(seed)


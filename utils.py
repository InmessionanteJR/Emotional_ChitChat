import numpy as np
import texar.torch as tx
from typing import List, Optional, Tuple

def get_perplexity(loss, round=2, base=2):
    if loss is None:
        return 0.
    return np.round(np.power(base, loss), round)

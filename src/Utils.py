import torch
import numpy as np
# ==============================
# Utils
# ==============================

def set_seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def huber_loss(pred, target, delta=1.0):
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    return (0.5 * quad**2 + delta * lin).mean()

def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    m, s = divmod(int(s + 0.5), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
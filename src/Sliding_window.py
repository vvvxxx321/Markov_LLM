import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple
# ==============================
# ETT Dataset loader (MM/MS)
# ==============================

class SlidingWindowETT(Dataset):
    """
    Loads ETTh1 and yields sliding windows in univariate-per-sample form for fair comparison:
    - Each sample selects one variable (feature) and returns:
      inputs X_past: (L, 1)
      target Y:      (H, 1)
    """
    def __init__(self, csv_path: str, seq_len: int, pred_len: int,
                 split: str = "train",
                 scale_by_train: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 target_mode: str = "MS",
                 target_index: int = -1):
        df = pd.read_csv(csv_path)
        # ETTh1 columns: ['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        data = df.values.astype(np.float32)         # (N, 7)
        Xall = data

        N = len(Xall)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.D_in = Xall.shape[1]

        # Time-LLM standard ETT-H split (12/4/4 months) with seq_len overlap for val/test
        use_ett_fixed = False
        ett_total = 12*30*24 + 8*30*24  # 14400
        if ("ETTh" in csv_path) or (N >= ett_total):
            use_ett_fixed = True

        if use_ett_fixed:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            split_idx = {"train": 0, "val": 1, "test": 2}[split]
            s, e = border1s[split_idx], border2s[split_idx]
            s = max(0, min(s, N))
            e = max(0, min(e, N))
            Xseg_raw = Xall[s:e]
            n_train = border2s[0]
        else:
            # percentage splits (≈60/20/20)
            n_train = int(0.6 * N)
            n_val   = int(0.2 * N)
            idx_map = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, N)}
            s, e = idx_map[split]
            Xseg_raw = Xall[s:e]

        # scale with TRAIN stats only
        if scale_by_train is None:
            mu = Xall[:n_train].mean(0, keepdims=True)
            sd = Xall[:n_train].std(0, keepdims=True) + 1e-6
        else:
            mu, sd = scale_by_train
        Xseg = (Xseg_raw - mu) / sd
        self.mu, self.sd = mu, sd

        # Determine target index
        tgt_idx = (self.D_in + target_index) if target_index < 0 else target_index
        tgt_idx = int(max(0, min(self.D_in - 1, tgt_idx)))

        # If target_mode == 'S', keep only the target feature (e.g., OT at index -1)
        if target_mode == 'S':
            self.Xseg = Xseg[:, [tgt_idx]]  # (T, 1)
            self.D_in = 1
        else:
            # Emit all features as separate univariate samples (Time-LLM style)
            self.Xseg = Xseg  # (T, D_in)
        self.D_out = 1
        self.tot_len = max(0, len(self.Xseg) - self.seq_len - self.pred_len + 1)

    def __len__(self):
        return self.tot_len * self.D_in

    def __getitem__(self, index):
        # Map to (feature_id, window_start)
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Univariate series for this feature
        x = self.Xseg[s_begin:s_end, feat_id:feat_id+1]           # (L, 1)
        y = self.Xseg[r_begin:r_end, feat_id:feat_id+1]           # (H, 1)
        t_idx = np.arange(s_begin, s_end, dtype=np.int64)         # (L,)
        return torch.from_numpy(x.astype(np.float32)), \
               torch.from_numpy(y.astype(np.float32)), \
               torch.from_numpy(t_idx)
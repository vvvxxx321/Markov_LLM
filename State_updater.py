import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class StateUpdater(nn.Module):
    def __init__(self, d_enc=256, d_state=512, periods=(24,168)):
        super().__init__()
        self.gru = nn.GRU(d_enc, d_state, batch_first=True)
        self.periods = periods
        self.season = nn.Linear(2*len(periods), d_state)
    def seasonal_feats(self, t_idx):
        B, L = t_idx.shape
        outs = []
        for p in self.periods:
            phi = 2*math.pi * t_idx.float() / p
            outs += [torch.sin(phi), torch.cos(phi)]
        return torch.stack(outs, dim=-1).view(B, L, -1)
    def forward(self, e, t_idx):
        seq, last = self.gru(e)  # last: (1,B,d_state)
        last = last.squeeze(0)
        seas = self.season(self.seasonal_feats(t_idx)[:, -1, :])
        return last + seas  # s_t
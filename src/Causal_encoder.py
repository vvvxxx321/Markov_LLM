import torch
import torch.nn as nn
import torch.nn.functional as F
# ==============================
# Causal encoder → state → prompts
# ==============================

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        pad = (k - 1) * dilation
        super().__init__(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self._trim = pad
    def forward(self, x):
        y = super().forward(x)
        if self._trim:
            y = y[..., :-self._trim]
        return y

class TCNBlock(nn.Module):
    def __init__(self, ch, k=3, dilation=1, pdrop=0.1, use_groupnorm: bool = False):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, k, dilation)
        self.c2 = CausalConv1d(ch, ch, k, dilation)
        if use_groupnorm:
            self.n1 = nn.GroupNorm(1, ch)
            self.n2 = nn.GroupNorm(1, ch)
        else:
            self.n1 = nn.BatchNorm1d(ch)
            self.n2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(pdrop)
    def forward(self, x):
        h = F.silu(self.n1(self.c1(x))); h = self.drop(h)
        h = self.n2(self.c2(h))
        return F.silu(x + h)

class CausalEncoder(nn.Module):
    def __init__(self, d_in, d_enc=256, layers=6, k=3, use_groupnorm: bool = False):
        super().__init__()
        self.inproj = nn.Conv1d(d_in, d_enc, 1)
        self.blocks = nn.ModuleList([TCNBlock(d_enc, k=k, dilation=2**i, use_groupnorm=use_groupnorm)
                                     for i in range(layers)])
    def forward(self, x):  # (B,L,D)
        h = x.transpose(1,2)
        h = self.inproj(h)
        for blk in self.blocks:
            h = blk(h)
        return h.transpose(1,2)  # (B,L,d_enc)
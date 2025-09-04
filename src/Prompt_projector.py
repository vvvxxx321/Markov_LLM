import torch
import torch.nn as nn
import torch.nn.functional as F
class PromptProjector(nn.Module):
    def __init__(self, d_state=512, d_llm=768, n_prompts=1):
        super().__init__()
        self.n_prompts = n_prompts
        self.net = nn.Sequential(
            nn.Linear(d_state, d_llm),
            nn.LayerNorm(d_llm),
            nn.SiLU(),
            nn.Linear(d_llm, d_llm)
        )
    def forward(self, s):  # (B,d_state)->(B,K,d_llm)
        B = s.shape[0]
        p = self.net(s).unsqueeze(1).repeat(1, self.n_prompts, 1)
        return p
    

class PromptProjectorMulti(nn.Module):
    def __init__(self, d_state: int, d_llm: int, n_prompts: int):
        super().__init__()
        self.n_prompts = int(n_prompts)
        self.ff = nn.Sequential(
            nn.Linear(d_state, d_llm),
            nn.LayerNorm(d_llm),
            nn.SiLU(),
            nn.Linear(d_llm, d_llm)
        )
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, K, D = S.shape
        return self.ff(S.view(B*K, D)).view(B, K, -1)
    

def segment_mean_pool(seq: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Mean-pool a (B,L,D) sequence into (B,n_segments,D)"""
    B, L, D = seq.shape
    if n_segments <= 1:
        return seq[:, -1:, :]
    edges = torch.linspace(0, L, n_segments+1, device=seq.device).floor().long()
    outs = []
    for i in range(n_segments):
        a, b = int(edges[i].item()), int(edges[i+1].item())
        if b <= a:
            outs.append(seq[:, -1:, :])
        else:
            outs.append(seq[:, a:b, :].mean(dim=1, keepdim=True))
    return torch.cat(outs, dim=1)
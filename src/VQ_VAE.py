import torch
import torch.nn as nn
import torch.nn.functional as F
# ==============================
# VQ-VAE style quantizer (learnable codebook)
# ==============================
class VectorQuantizer(nn.Module):
    """
    Simple VQ layer with straight-through estimator.
    Codebook E: (K, D). Given z_e: (N, D) → indices (N,), z_q: (N, D).
    Losses: codebook + commitment.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e: torch.Tensor):
        # z_e: (N, D)
        with torch.no_grad():
            # Compute distances ||z - e||^2 = z^2 + e^2 - 2 z·e
            z_e_sq = (z_e**2).sum(dim=1, keepdim=True)                 # (N,1)
            e_sq = (self.codebook.weight**2).sum(dim=1).unsqueeze(0)   # (1,K)
            distances = z_e_sq + e_sq - 2.0 * (z_e @ self.codebook.weight.t())
            indices = distances.argmin(dim=1)                          # (N,)
        z_q = self.codebook(indices)                                   # (N,D)
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        # Losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        return z_q_st, indices, codebook_loss, commitment_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
class TrueMarkovChain(nn.Module):
    """True Markov Chain where P(S(t) | S(t-1), X(t)) = P(S(t) | S(1:t-1), X(1:t))"""
    def __init__(self, d_in: int, d_state: int):
        super().__init__()
        self.d_in = d_in
        self.d_state = d_state
        
        # State initialization from first observation: S(0) = g(X(0))
        self.init_proj = nn.Sequential(
            nn.Linear(d_in, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state)
        )
        
        # True Markov transition: S(t) = f(S(t-1), X(t))
        self.transition = nn.Sequential(
            nn.Linear(d_state + d_in, d_state * 2),
            nn.SiLU(),
            nn.Linear(d_state * 2, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state)
        )
        
        # Observation model for enforcing meaningful states: X̂(t) = h(S(t))
        self.obs_model = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_in)
        )
        
    def forward(self, past: torch.Tensor):
        """
        Build true Markov states where each S(t) only depends on S(t-1) and X(t)
        Args:
            past: (B, L, d_in) - input sequence
        Returns:
            states: (B, L, d_state) - Markov states
        """
        B, L, D_in = past.shape
        
        # Initialize: S(0) = g(X(0))
        s_prev = self.init_proj(past[:, 0, :])
        states = [s_prev]
        
        # Markov transitions: S(t) = f(S(t-1), X(t)) for t = 1, ..., L-1
        for t in range(1, L):
            s_input = torch.cat([s_prev, past[:, t, :]], dim=-1)
            s_curr = self.transition(s_input)
            states.append(s_curr)
            s_prev = s_curr
            
        return torch.stack(states, dim=1)  # (B, L, d_state)
    
    def compute_markov_loss(self, past: torch.Tensor):
        """
        Enforce Markov property by predicting observations from states
        This ensures states contain predictively useful information
        """
        if past.size(1) <= 1:
            return torch.tensor(0.0, device=past.device)
            
        # Get states for X(0:T-1)
        states = self.forward(past[:, :-1, :])  # (B, L-1, d_state)
        
        # Predict X(1:T) from S(0:T-1) - ensures states are meaningful
        pred_obs = self.obs_model(states)  # (B, L-1, d_in)
        target_obs = past[:, 1:, :]  # (B, L-1, d_in)
        
        return F.mse_loss(pred_obs, target_obs)
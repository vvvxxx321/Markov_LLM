import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
@dataclass
class Config:
    d_in: int = 7
    d_out: int = 1          # MS default
    seq_len: int = 512
    pred_len: int = 96
    patch: int = 16         # => M=pred_len/patch steps
    K_codes: int = 512
    d_enc: int = 256
    d_state: int = 512
    d_llm: int = 768        # GPT-2 hidden size
    n_prompts: int = 1
    hf_model_name: str = "gpt2"
    lr: float = 3e-4
    freeze_llm: bool = True
    d_latent: int = 128      # VQ latent dim for patch tokens
    vq_beta: float = 0.25    # commitment cost
    use_groupnorm: bool = False
    label_smoothing: float = 0.05



    @torch.no_grad()
    def sample(self, past, t_idx, S=100, top_p=0.9, temperature=1.0):
        """
        Returns numeric samples: (S, B, pred_len, D_out)
        Vectorized across S by tiling batch (no KV cache for simplicity here).
        """
        B = past.size(0)
        device = past.device
        e = self.encoder(past); s = self.state(e, t_idx)
        prompts = self.proj(s)                                       # (B,1,d_llm)
        # Tile across S
        prompts = prompts.repeat_interleave(S, dim=0)                # (S*B,1,d_llm)
        step_ids_all = torch.arange(self.M, device=device).unsqueeze(0).repeat(B,1)
        step_ids_all = step_ids_all.repeat_interleave(S, dim=0)      # (S*B, M)

        cb = self.quantizer.codebook.weight                          # (K, d_latent)
        gen_embs = [self.bos.expand(B*S,1,-1)]
        codes_list = []
        for t in range(self.M):
            prev_seq = torch.cat(gen_embs, dim=1)                    # (S*B, t+1, d_llm)
            step_emb = self.step_embed(step_ids_all[:, :t+1])        # (S*B, t+1, d_llm)
            step_in  = prev_seq + step_emb
            inputs_embeds = torch.cat([prompts, step_in], dim=1)
            attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

            h = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask).last_hidden_state
            h_t = h[:, -1, :]
            logits = self.head(h_t) / temperature                    # (S*B,K)
            probs = F.softmax(logits, dim=-1)
            sp, si = probs.sort(dim=-1, descending=True)             # (S*B,K)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            sp = sp / sp.sum(dim=-1, keepdim=True)
            idx_rank = torch.multinomial(sp, 1)                      # (S*B,1)
            idx = si.gather(1, idx_rank).squeeze(1)                  # (S*B,)
            codes_list.append(idx.view(B*S, 1))

            lat = cb.index_select(0, idx)                             # (S*B, d_latent)
            gen_embs.append(self.code_to_llm(lat).unsqueeze(1))       # (S*B,1,d_llm)

        codes = torch.cat(codes_list, dim=1).view(S, B, self.M)      # (S,B,M)
        patches = self._recon_from_codes(codes).view(S, B, self.M, -1)
        patches = patches.view(S, B, -1, self.cfg.d_out)             # (S,B,H,D_out)
        return patches
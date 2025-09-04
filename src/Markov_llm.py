import torch
import torch.nn as nn
import torch.nn.functional as F
from Causal_encoder import CausalEncoder
from State_updater import StateUpdater
from transformers import GPT2Model
from VQ_VAE import VectorQuantizer
from Utils import huber_loss
from Prompt_projector import segment_mean_pool , PromptProjector , PromptProjectorMulti
from Markov_chain import MarkovChain
from ExptConfig import ExptConfig
from Config import Config



class MarkovLLM_GPT2(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert cfg.pred_len % cfg.patch == 0, "pred_len must be divisible by patch"
        self.M = cfg.pred_len // cfg.patch

        self.encoder = CausalEncoder(cfg.d_in, cfg.d_enc, layers=6, use_groupnorm=cfg.use_groupnorm)
        self.state   = StateUpdater(cfg.d_enc, cfg.d_state)
        self.proj    = PromptProjector(cfg.d_state, cfg.d_llm, cfg.n_prompts)

        # GPT-2 backbone (frozen by default)
        self.gpt2 = GPT2Model.from_pretrained(cfg.hf_model_name)
        if cfg.freeze_llm:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        # AR scaffolding
        self.step_embed = nn.Embedding(self.M, cfg.d_llm)
        self.bos = nn.Parameter(torch.randn(1,1,cfg.d_llm)*0.02)

        # Token head + VQ-VAE style patch codec
        self.head = nn.Linear(cfg.d_llm, cfg.K_codes)
        self.patch_encoder = nn.Sequential(
            nn.Linear(cfg.patch*cfg.d_out, cfg.d_latent),
            nn.SiLU(),
            nn.Linear(cfg.d_latent, cfg.d_latent)
        )
        self.quantizer = VectorQuantizer(cfg.K_codes, cfg.d_latent, commitment_cost=cfg.vq_beta)
        self.patch_decoder = nn.Sequential(
            nn.Linear(cfg.d_latent, 256),
            nn.SiLU(),
            nn.Linear(256, cfg.patch*cfg.d_out)
        )
        # Map VQ latent → LLM embedding for teacher forcing / feedback
        self.code_to_llm = nn.Linear(cfg.d_latent, cfg.d_llm)

    def _recon_from_codes(self, code_indices: torch.Tensor) -> torch.Tensor:
        # code_indices: (...,)
        codes_latent = self.quantizer.codebook(code_indices.view(-1))            # (N,D_latent)
        recon = self.patch_decoder(codes_latent).view(*code_indices.shape, -1)   # (..., patch*d_out)
        return recon

    def forward(self, past, t_idx, target_codes=None, target_patches=None,
                tie_weight=0.1, vq_weight: float = 1.0):
        """
        past: (B,L,D_in)
        t_idx:(B,L)
        target_codes:  (B,M) optional; if None we quantize target_patches
        target_patches:(B,M,patch*D_out) numeric patches for tie losses
        """
        B = past.size(0)
        e = self.encoder(past)
        s = self.state(e, t_idx)
        prompts = self.proj(s)                                  # (B,1,d_llm)

        # Derive codes from numeric target via VQ if not provided
        cb_loss = com_loss = None
        if (target_codes is None) and (target_patches is not None):
            M = target_patches.size(1)
            y_flat = target_patches.reshape(B*M, -1)            # (B*M, P)
            z_e = self.patch_encoder(y_flat)                    # (B*M, d_latent)
            z_q, indices, cb_loss, com_loss = self.quantizer(z_e)
            target_codes = indices.view(B, M)                   # (B,M)
        elif target_codes is None:
            raise AssertionError("Either target_codes or target_patches must be provided.")

        # Teacher forcing tokens from VQ LATENTS (aligned with decoder)
        M = target_codes.size(1)
        if M != self.M:
            raise AssertionError(f"target_codes has M={M} but model expects {self.M}.")
        prev_lat = self.quantizer.codebook(target_codes[:, :-1].reshape(-1)).detach() \
                      .view(B, M-1, -1)                         # (B,M-1,d_latent)
        prev_tok = torch.cat([self.bos.expand(B,1,-1),
                              self.code_to_llm(prev_lat)], dim=1) # (B,M,d_llm)

        # add step embeddings
        step_ids = torch.arange(self.M, device=past.device).unsqueeze(0).repeat(B,1)
        step_in  = prev_tok + self.step_embed(step_ids)         # (B,M,d_llm)

        # LLM pass
        inputs_embeds = torch.cat([prompts, step_in], dim=1)    # (B,1+M,d_llm)
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=past.device)
        out = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask).last_hidden_state
        h_steps = out[:, prompts.shape[1]:, :]                  # (B,M,d_llm)
        logits = self.head(h_steps)                             # (B,M,K)

        # Losses
        losses = {}
        losses["ce"] = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_codes.view(-1),
            label_smoothing=self.cfg.label_smoothing
        )

        if target_patches is not None:
            # 1) VQ codebook + commitment (only when we quantized this batch)
            if cb_loss is not None:
                losses["vq_cb"] = cb_loss * vq_weight
                losses["vq_commit"] = com_loss * vq_weight

            # 2) Hard-tie: reconstruct numeric target from its quantized latent
            if cb_loss is not None:
                z_q_dec = self.patch_decoder(z_q).view(B, -1, self.cfg.patch*self.cfg.d_out)  # (B,M,P)
                recon_num = z_q_dec
            else:
                recon_num = self._recon_from_codes(target_codes)                               # (B,M,P)
            losses["tie"] = huber_loss(recon_num, target_patches) * tie_weight

            # 3) Soft-tie: decode expected patch under current logits
            probs = F.softmax(logits, dim=-1)                                                 # (B,M,K)
            K = self.cfg.K_codes
            code_ids = torch.arange(K, device=logits.device)
            code_recon = self.patch_decoder(self.quantizer.codebook(code_ids))                # (K,P)
            exp_patch = torch.matmul(probs, code_recon)                                       # (B,M,P)
            losses["soft_tie"] = huber_loss(exp_patch, target_patches) * (tie_weight * 0.5)

        return logits, losses
    


class MarkovLLM_GPT2_Expt(nn.Module):
    def __init__(self, cfg: ExptConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.pred_len % cfg.patch == 0, "pred_len must be divisible by patch"
        self.M = cfg.pred_len // cfg.patch

        self.encoder = CausalEncoder(cfg.d_in, cfg.d_enc, layers=6, use_groupnorm=cfg.use_groupnorm)
        self.state   = StateUpdater(cfg.d_enc, cfg.d_state)
        # Markov chain that works on encoded features for fair comparison
        self.markov = MarkovChain(cfg.d_enc, cfg.d_state)

        self.proj_1 = PromptProjector(cfg.d_state, cfg.d_llm, n_prompts=1)
        self.proj_k = PromptProjectorMulti(cfg.d_state, cfg.d_llm, n_prompts=cfg.n_prompts)

        self.gpt2 = GPT2Model.from_pretrained(cfg.hf_model_name)
        if cfg.freeze_llm:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        # Add missing components from original model
        self.step_embed = nn.Embedding(self.M, cfg.d_llm)
        self.bos = nn.Parameter(torch.randn(1,1,cfg.d_llm)*0.02)
        self.head = nn.Linear(cfg.d_llm, cfg.K_codes)

        self.patch_encoder = nn.Sequential(
            nn.Linear(cfg.patch*cfg.d_out, cfg.d_latent),
            nn.SiLU(),
            nn.Linear(cfg.d_latent, cfg.d_latent)
        )
        self.quantizer = VectorQuantizer(cfg.K_codes, cfg.d_latent, commitment_cost=cfg.vq_beta)
        self.patch_decoder = nn.Sequential(
            nn.Linear(cfg.d_latent, 256),
            nn.SiLU(),
            nn.Linear(256, cfg.patch*cfg.d_out)
        )
        self.code_to_llm = nn.Linear(cfg.d_latent, cfg.d_llm)

        self.raw_cnn = nn.Sequential(
            nn.Conv1d(cfg.d_in, cfg.d_state, kernel_size=5, padding=4, dilation=2),
            nn.SiLU(),
            nn.Conv1d(cfg.d_state, cfg.d_state, kernel_size=5, padding=8, dilation=4),
            nn.SiLU(),
        )

    def _states_cur(self, e: torch.Tensor, t_idx: torch.Tensor):
        # Return (seq, last) from current GRU + season
        seq, last = self.state.gru(e)
        last = last.squeeze(0)
        seq_seas = self.state.season(self.state.seasonal_feats(t_idx))
        seas = seq_seas.mean(dim=1)
        # Avoid in-place operations
        seq_out = seq.add(seq_seas)
        last_out = last.add(seas)
        return seq_out, last_out

    # (Removed) legacy mc_* states builder

    def _prep_prompts(self, past: torch.Tensor, t_idx: torch.Tensor):
        cfg = self.cfg
        losses = {}
        
        # ALL modes start with the same CausalEncoder for fair comparison
        e = self.encoder(past)  # (B, L, d_enc)
        
        if cfg.history_mode == "gru_last":
            s = self.state(e, t_idx)  # Use the original StateUpdater.forward() 
            prompts = self.proj_1(s)
        elif cfg.history_mode in ("markov_last", "markov_all"):
            # Markov Chain on ENCODED features for fair comparison
            # P(S(t) | S(t-1), E(t)) = P(S(t) | S(1:t-1), E(1:t))
            seq_mc = self.markov(e)  # (B, L, d_state)
            
            # Add Markov loss to enforce meaningful state representations
            L_mc = self.markov.compute_markov_loss(e)
            losses["loss_markov"] = cfg.mc_weight * L_mc
            
            if cfg.history_mode == "markov_last":
                prompts = self.proj_1(seq_mc[:, -1, :])  # Use last Markov state
            else:  # markov_all
                n_seg = seq_mc.size(1) if (self.cfg.n_prompts <= 0) else min(self.cfg.n_prompts, seq_mc.size(1))
                pooled = segment_mean_pool(seq_mc, n_seg)
                prompts = self.proj_k(pooled)  # Use all Markov states (or as many as allowed)
        else:
            # Other modes use encoded features with GRU processing
            seq_cur, last_cur = self._states_cur(e, t_idx)
            if cfg.history_mode == "gru_all":
                n_seg = seq_cur.size(1) if (self.cfg.n_prompts <= 0) else min(self.cfg.n_prompts, seq_cur.size(1))
                pooled = segment_mean_pool(seq_cur, n_seg)
                prompts = self.proj_k(pooled)
            # (Removed) legacy mc_* modes
            elif cfg.history_mode == "raw_all":
                feat = self.raw_cnn(past.transpose(1,2)).transpose(1,2)
                n_seg = feat.size(1) if (self.cfg.n_prompts <= 0) else min(self.cfg.n_prompts, feat.size(1))
                pooled = segment_mean_pool(feat, n_seg)
                prompts = self.proj_k(pooled)
            else:
                raise ValueError(f"Unknown history_mode: {cfg.history_mode}")
        return prompts, losses

    def forward(self, past, t_idx, target_codes=None, target_patches=None, tie_weight=0.1, vq_weight: float = 1.0):
        B = past.size(0); M = self.M
        prompts, extra = self._prep_prompts(past, t_idx)

        # ALL MODES use the original baseline architecture - only prompts differ
        # Derive codes from numeric target via VQ if not provided
        cb_loss = com_loss = None
        if (target_codes is None) and (target_patches is not None):
            y_flat = target_patches.reshape(B*M, -1)
            z_e = self.patch_encoder(y_flat)
            z_q, indices, cb_loss, com_loss = self.quantizer(z_e)
            target_codes = indices.view(B, M)
        elif target_codes is None:
            raise AssertionError("Either target_codes or target_patches must be provided.")

        # Teacher forcing tokens from VQ LATENTS (aligned with decoder) - ORIGINAL APPROACH
        if M != self.M:
            raise AssertionError(f"target_codes has M={M} but model expects {self.M}.")
        
        # Handle M=1 case (no previous tokens)
        if M == 1:
            prev_tok = self.bos.expand(B, 1, -1)  # Only BOS token
        else:
            prev_lat = self.quantizer.codebook(target_codes[:, :-1].reshape(-1)).detach().view(B, M-1, -1)
            prev_tok = torch.cat([self.bos.expand(B,1,-1), self.code_to_llm(prev_lat)], dim=1)

        # add step embeddings
        step_ids = torch.arange(self.M, device=past.device).unsqueeze(0).repeat(B,1)
        step_in = prev_tok + self.step_embed(step_ids)

        # LLM pass
        inputs_embeds = torch.cat([prompts, step_in], dim=1)
        attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=past.device)
        out = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask).last_hidden_state
        h_steps = out[:, prompts.shape[1]:, :]
        logits = self.head(h_steps)

        # Losses - ORIGINAL APPROACH (same for all modes)
        losses = {}
        losses["ce"] = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_codes.view(-1),
            label_smoothing=0.05  # Use same as original
        )

        if target_patches is not None:
            # 1) VQ codebook + commitment (only when we quantized this batch)
            if cb_loss is not None:
                losses["vq_cb"] = cb_loss * vq_weight
                losses["vq_commit"] = com_loss * vq_weight

            # 2) Hard-tie: reconstruct numeric target from its quantized latent
            if cb_loss is not None:
                z_q_dec = self.patch_decoder(z_q).view(B, -1, self.cfg.patch*self.cfg.d_out)
                recon_num = z_q_dec
            else:
                codes_latent = self.quantizer.codebook(target_codes.view(-1))
                recon_num = self.patch_decoder(codes_latent).view(B, M, -1)
            losses["tie"] = huber_loss(recon_num, target_patches) * tie_weight

            # 3) Soft-tie: decode expected patch under current logits
            probs = F.softmax(logits, dim=-1)
            K = self.cfg.K_codes
            code_ids = torch.arange(K, device=logits.device)
            code_recon = self.patch_decoder(self.quantizer.codebook(code_ids))
            exp_patch = torch.matmul(probs, code_recon)
            losses["soft_tie"] = huber_loss(exp_patch, target_patches) * (tie_weight * 0.5)

        # Add Markov chain losses
        if "loss_mc" in extra:
            losses["mc"] = extra["loss_mc"]
        if "loss_markov" in extra:
            losses["markov"] = extra["loss_markov"]

        return logits, losses

    @torch.no_grad()
    def sample(self, past, t_idx, S: int = 32, top_p: float = 0.9, temperature: float = 1.0):
        device = past.device
        B = past.size(0); M = self.M
        prompts, _ = self._prep_prompts(past, t_idx)
        
        # ALL MODES use the original baseline sampling approach - only prompts differ
        prompts = prompts.repeat_interleave(S, dim=0)  # (S*B, n_prompts, d_llm)
        step_ids_all = torch.arange(self.M, device=device).unsqueeze(0).repeat(B,1)
        step_ids_all = step_ids_all.repeat_interleave(S, dim=0)  # (S*B, M)

        cb = self.quantizer.codebook.weight  # (K, d_latent)
        gen_embs = [self.bos.expand(B*S,1,-1)]
        codes_list = []
        for t in range(self.M):
            prev_seq = torch.cat(gen_embs, dim=1)  # (S*B, t+1, d_llm)
            step_emb = self.step_embed(step_ids_all[:, :t+1])  # (S*B, t+1, d_llm)
            step_in = prev_seq + step_emb
            inputs_embeds = torch.cat([prompts, step_in], dim=1)
            attn_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=device)

            h = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask).last_hidden_state
            h_t = h[:, -1, :]
            logits = self.head(h_t) / temperature  # (S*B,K)
            probs = F.softmax(logits, dim=-1)
            sp, si = probs.sort(dim=-1, descending=True)  # (S*B,K)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            sp = sp / sp.sum(dim=-1, keepdim=True)
            idx_rank = torch.multinomial(sp, 1)  # (S*B,1)
            idx = si.gather(1, idx_rank).squeeze(1)  # (S*B,)
            codes_list.append(idx.view(B*S, 1))

            lat = cb.index_select(0, idx)  # (S*B, d_latent)
            gen_embs.append(self.code_to_llm(lat).unsqueeze(1))  # (S*B,1,d_llm)

        codes = torch.cat(codes_list, dim=1).view(S, B, self.M)  # (S,B,M)
        codes_latent = self.quantizer.codebook(codes.view(-1))
        patches = self.patch_decoder(codes_latent).view(S, B, self.M, -1)
        patches = patches.view(S, B, -1, self.cfg.d_out)  # (S,B,H,D_out)
        return patches
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import GPT2Model

from Markov_llm import MarkovLLM_GPT2, MarkovLLM_GPT2_Expt
from Early_stop import EarlyStopper
from Sliding_window import SlidingWindowETT
from Causal_encoder import CausalEncoder
from Utils import set_seed, _fmt_secs
from ExptConfig import ExptConfig
from Config import Config


@torch.no_grad()
def crps_from_samples(y_true: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """
    y_true:   (B, H*D)  — flattened numeric future
    samples:  (S, B, H*D)
    """
    term1 = (samples - y_true.unsqueeze(0)).abs().mean()
    diffs = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean()
    return term1 - 0.5 * diffs
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


# ==============================
# Markovized LLM (VQ-token outputs)
# ==============================


def build_expt_model(args, d_in, d_out):
    cfg = ExptConfig(
        d_in=d_in, d_out=d_out,
        seq_len=args.seq_len, pred_len=args.pred_len, patch=args.patch,
        d_enc=args.d_enc, d_state=args.d_state, use_groupnorm=args.use_groupnorm,
        hf_model_name=args.hf_model_name, d_llm=args.d_llm, n_prompts=args.n_prompts,
        freeze_llm=args.freeze_llm,
        K_codes=args.K_codes, d_latent=args.d_latent, vq_beta=args.vq_beta,
        history_mode=args.expt_mode, downsample=args.downsample, mc_weight=args.mc_weight,
    )
    return MarkovLLM_GPT2_Expt(cfg)

# ==============================
# Train / Eval helpers
# ==============================

def train_epoch(model, opt, train_loader, pred_len, d_out, patch, device,
                epoch_idx: int = 1, epochs: int = 1,
                global_start: float = None, total_batches: int = None,
                log_interval: int = 50):
    model.train()
    tot = 0.0
    n_batches = len(train_loader)
    if global_start is None:
        global_start = time.time()
    for i, (xb, yb, tb) in enumerate(train_loader, 1):
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        B = xb.size(0); M = pred_len // patch
        y_patches = yb.view(B, M, patch*d_out)
        logits, losses = model(xb, tb, target_codes=None, target_patches=y_patches, tie_weight=0.2, vq_weight=1.0)
        loss = sum(losses.values())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()

        if (i % max(1, log_interval) == 0) or (i == n_batches):
            global_done = (epoch_idx - 1) * n_batches + i
            if total_batches is None:
                total_batches = epochs * n_batches
            elapsed = time.time() - global_start
            avg_per_batch = elapsed / max(1, global_done)
            remain_total = (total_batches - global_done) * avg_per_batch
            remain_epoch = (n_batches - i) * avg_per_batch
            print(f"Epoch {epoch_idx:02d}/{epochs} | Batch {i:04d}/{n_batches} "
                  f"| Elapsed { _fmt_secs(elapsed) } | ETA epoch { _fmt_secs(remain_epoch) } "
                  f"| ETA total { _fmt_secs(remain_total) }", flush=True)
    return tot / max(1, len(train_loader))

@torch.no_grad()
def eval_epoch(model, data_loader, pred_len, d_out, patch, device, S=32, top_p=0.9, temperature=1.0):
    model.eval()
    tot_ce, tot_tie, n = 0.0, 0.0, 0
    crps_vals, mae_vals, mse_vals = [], [], []
    for xb, yb, tb in data_loader:
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
        B = xb.size(0); M = pred_len // patch
        # No teacher forcing for predictions at eval/test: sample S independent paths
        samp = model.sample(xb, tb, S=S, top_p=top_p, temperature=temperature)   # (S,B,H,D_out)
        y_true = yb
        # CRPS from samples
        crps_vals.append(crps_from_samples(y_true.view(B, -1), samp.view(S, B, -1)).item())
        # MC mean as point prediction for MAE/MSE
        exp = samp.mean(dim=0)  # (B,H,D_out)
        mae_vals.append(torch.mean(torch.abs(exp - y_true)).item())
        mse_vals.append(torch.mean((exp - y_true)**2).item())
        # For logging CE/tie consistently, we can still compute them teacher-forced on targets
        y_patches = yb.view(B, M, patch*d_out)
        _, losses = model(xb, tb, target_codes=None, target_patches=y_patches, tie_weight=0.2)
        tot_ce += float(losses.get("ce", 0.0))
        tot_tie += float(losses.get("tie", 0.0)) + float(losses.get("soft_tie", 0.0))
        n += 1
    return {
        "ce": tot_ce/max(1,n),
        "tie": tot_tie/max(1,n),
        "crps": float(np.mean(crps_vals)) if crps_vals else 0.0,
        "mae": float(np.mean(mae_vals)) if mae_vals else 0.0,
        "mse": float(np.mean(mse_vals)) if mse_vals else 0.0,
    }

# ==============================
# Main with Early Stopping
# ==============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch", type=int, default=12)  # use 12 for "2 patches = 1 day" on hourly ETT
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--K_codes", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_model_name", type=str, default="gpt2")
    parser.add_argument("--freeze_llm", action="store_true")
    parser.add_argument("--no-freeze_llm", dest="freeze_llm", action="store_false")
    parser.add_argument("--target_mode", type=str, default="MS", choices=["MM","MS", "S"])
    parser.add_argument("--target_index", type=int, default=-1)
    parser.add_argument("--eval_S", type=int, default=32)
    parser.add_argument("--eval_top_p", type=float, default=0.9)
    parser.add_argument("--eval_temp", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--use_groupnorm", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    
    # Experiment variants
    parser.add_argument("--expt_mode", type=str, default="none", 
                        choices=["none","gru_last","gru_all","raw_all","markov_last","markov_all"])
    # n_prompts semantics:
    #   > 0  => use up to n_prompts pooled segments (clamped to T)
    #   <= 0 => use all available states T (no pooling)
    parser.add_argument("--n_prompts", type=int, default=1)
    parser.add_argument("--d_enc", type=int, default=256)
    parser.add_argument("--d_state", type=int, default=512)
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--d_latent", type=int, default=128)
    parser.add_argument("--vq_beta", type=float, default=0.25)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--mc_weight", type=float, default=0.1)

    # Early stop controls
    parser.add_argument("--early_metric", type=str, default="mae", choices=["mae","mse","crps","ce"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--save_path", type=str, default="./best_markov_llm_vq.pt")

    parser.set_defaults(freeze_llm=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = SlidingWindowETT(args.csv_path, args.seq_len, args.pred_len,
                                split="train", target_mode=args.target_mode, target_index=args.target_index)
    val_ds   = SlidingWindowETT(args.csv_path, args.seq_len, args.pred_len,
                                split="val",  scale_by_train=(train_ds.mu, train_ds.sd),
                                target_mode=args.target_mode, target_index=args.target_index)
    test_ds  = SlidingWindowETT(args.csv_path, args.seq_len, args.pred_len,
                                split="test", scale_by_train=(train_ds.mu, train_ds.sd),
                                target_mode=args.target_mode, target_index=args.target_index)

    d_in  = train_ds.D_in
    d_out = train_ds.D_out
    assert args.pred_len % args.patch == 0, "pred_len must be divisible by patch"

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model selection: baseline (none) vs experiment variants
    if args.expt_mode == "none":
        cfg = Config(d_in=d_in, d_out=d_out, seq_len=args.seq_len, pred_len=args.pred_len, patch=args.patch,
                     K_codes=args.K_codes, d_llm=768, hf_model_name=args.hf_model_name,
                     lr=args.lr, freeze_llm=args.freeze_llm,
                     use_groupnorm=args.use_groupnorm, label_smoothing=args.label_smoothing)
        model = MarkovLLM_GPT2(cfg).to(device)
    else:
        model = build_expt_model(args, d_in=d_in, d_out=d_out).to(device)
        # Create a dummy cfg for experiment mode (needed for optimizer and checkpointing)
        cfg = Config(d_in=d_in, d_out=d_out, seq_len=args.seq_len, pred_len=args.pred_len, patch=args.patch,
                     K_codes=args.K_codes, d_llm=768, hf_model_name=args.hf_model_name,
                     lr=args.lr, freeze_llm=args.freeze_llm,
                     use_groupnorm=args.use_groupnorm, label_smoothing=args.label_smoothing)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )

    # Early stopper
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, mode="min")
    best_epoch = 0

    global_start = time.time()
    total_batches = args.epochs * len(train_dl)

    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, opt, train_dl, args.pred_len, d_out, args.patch, device,
                              epoch_idx=epoch, epochs=args.epochs,
                              global_start=global_start, total_batches=total_batches,
                              log_interval=args.log_interval)
        val_metrics = eval_epoch(model, val_dl, args.pred_len, d_out, args.patch, device,
                                 S=max(16, args.eval_S//2),
                                 top_p=args.eval_top_p, temperature=args.eval_temp)

        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.3f} | "
              f"val_ce {val_metrics['ce']:.3f} | val_tie {val_metrics['tie']:.3f} | "
              f"val_mae {val_metrics['mae']:.3f} | val_mse {val_metrics['mse']:.3f} | "
              f"val_crps {val_metrics['crps']:.4f}")

        score = val_metrics[args.early_metric]
        improved, stop = stopper.step(score)
        if improved:
            best_epoch = epoch
            torch.save({"cfg": cfg, "state_dict": model.state_dict()}, args.save_path)
            print(f"[checkpoint] Saved best {args.early_metric}={score:.4f} at epoch {epoch} → {args.save_path}")
        if stop:
            print(f"[early stop] No improvement in {args.patience} epochs. Stopping at epoch {epoch}.")
            break


    # Load best checkpoint for test (support PyTorch 2.6 safe loading and both dict/state_dict formats)
    try:
        ckpt = None
        try:
            # Try loading full object (dict with cfg/state_dict)
            import torch.serialization as ts
            if hasattr(ts, "add_safe_globals"):
                ts.add_safe_globals([Config])
            ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
        except Exception:
            # Fall back to weights-only (pure state_dict or dict)
            ckpt = torch.load(args.save_path, map_location=device, weights_only=True)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[load best] Loaded checkpoint from epoch {best_epoch} for test.")
    except Exception as e:
        print(f"[warn] Could not load checkpoint: {e}")

        # Test
    test_metrics = eval_epoch(model, test_dl, args.pred_len, d_out, args.patch, device,
                              S=args.eval_S, top_p=args.eval_top_p, temperature=args.eval_temp)
    print("TEST:", test_metrics)

if __name__ == "__main__":
    main()
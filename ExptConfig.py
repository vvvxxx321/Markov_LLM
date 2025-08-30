from dataclasses import dataclass
@dataclass
class ExptConfig:
    d_in: int = 1
    d_out: int = 1
    seq_len: int = 512
    pred_len: int = 96
    patch: int = 16
    d_enc: int = 256
    d_state: int = 512
    use_groupnorm: bool = False
    hf_model_name: str = "gpt2"
    d_llm: int = 768
    n_prompts: int = 1
    freeze_llm: bool = True
    K_codes: int = 512
    d_latent: int = 128
    vq_beta: float = 0.25
    history_mode: str = "cur_last"  # cur_last, cur_all, raw_all, true_markov_last, true_markov_all
    downsample: int = 8
    mc_weight: float = 0.1
set -e

CSV="./dataset/ETTh1.csv"
MODE="markov_last"      # gru_last | gru_all | markov_last | markov_all | raw_all

python -u ./src/run_main.py \
  --csv_path "$CSV" \
  --seq_len 512 --pred_len 96 --patch 12 \
  --batch_size 64 --epochs 50 \
  --K_codes 512 --hf_model_name gpt2 --freeze_llm \
  --target_mode S --target_index -1 --use_groupnorm \
  --eval_S 4 --eval_top_p 0.85 --eval_temp 0.8 \
  --patience 5 --early_metric mse --min_delta 0.001 \
  --expt_mode "$MODE" --n_prompts 1 \
  --mc_weight 0.15
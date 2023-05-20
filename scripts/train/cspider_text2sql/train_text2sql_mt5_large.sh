set -e

# train text2sql-mt5-large (CSpider version) model
python -u text2sql.py \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --device "0" \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-cspider" \
    --tensorboard_save_path "./tensorboard_log/text2sql-mt5-large-cspider" \
    --model_name_or_path "google/mt5-large" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_cspider.json"

# select the best text2sql-mt5-large (CSpider version) ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size 12 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-cspider" \
    --eval_results_path "./eval_results/text2sql-mt5-large-cspider" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev_cspider.json" \
    --original_dev_filepath "./data/CSpider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "sql"
set -e

# train text2natsql-mt5-base-cspider model
python -u text2sql.py \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --device "0" \
    --learning_rate 1e-4 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2natsql-mt5-base-cspider" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-mt5-base-cspider" \
    --model_name_or_path "google/mt5-base" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_cspider_natsql.json"

# select the best text2natsql-mt5-base-cspider ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size 32 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2natsql-mt5-base-cspider" \
    --eval_results_path "./eval_results/text2natsql-mt5-base-cspider" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev_cspider_natsql.json" \
    --original_dev_filepath "./data/CSpider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/cspider_tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql"
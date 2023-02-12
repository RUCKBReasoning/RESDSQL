set -e

# train text2sql-t5-3b model
python -u text2sql.py \
    --batch_size 6 \
    --gradient_descent_step 16 \
    --device "0" \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2sql-t5-3b" \
    --tensorboard_save_path "./tensorboard_log/text2sql-t5-3b" \
    --model_name_or_path "t5-3b" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider.json"

# select the best text2sql-t5-3b ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size 2 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-t5-3b" \
    --eval_results_path "./eval_results/text2sql-t5-3b" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "sql"
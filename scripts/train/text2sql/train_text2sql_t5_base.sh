set -e

# train text2sql-t5-base model
python -u text2sql.py \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --learning_rate 1e-4 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2sql-t5-base-additional" \
    --tensorboard_save_path "./tensorboard_log/text2sql-t5-base-additional" \
    --model_name_or_path "./llm/t5-base" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train.json"
    
# select the best text2sql-t5-base ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size 32 \
    --device "7" \
    --seed 42 \
    --save_path "./models/text2sql-t5-base-additional" \
    --eval_results_path "./eval_results/text2sql-t5-base-additional" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "sql"
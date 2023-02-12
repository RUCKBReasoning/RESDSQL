set -e

device="0"

python -u evaluate_text2sql_ckpts.py \
    --batch_size 3 \
    --device $device \
    --seed 42 \
    --save_path "./models/text2natsql-t5-3b" \
    --eval_results_path "./eval_results/text2natsql-t5-3b-spider-syn" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_spider_syn_natsql.json" \
    --original_dev_filepath "./data/spider-syn/dev_syn.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/spider_syn_tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql"
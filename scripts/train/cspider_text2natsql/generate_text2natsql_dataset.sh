set -e

# generate text2natsql training dataset with noise_rate 0.2
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_train_cspider_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_train_cspider_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "train" \
    --noise_rate 0.2 \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# predict probability for each schema item in the eval set
python schema_item_classifier.py \
    --batch_size 32 \
    --device "0" \
    --seed 42 \
    --save_path "./models/xlm_roberta_text2natsql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_dev_cspider_natsql.json" \
    --output_filepath "./data/preprocessed_data/dev_cspider_with_probs_natsql.json" \
    --use_contents \
    --mode "eval"

# generate text2natsql development dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/dev_cspider_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_dev_cspider_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "eval" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"
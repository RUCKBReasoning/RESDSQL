set -e

device="0"
tables_for_natsql="./data/preprocessed_data/spider_dk_tables_for_natsql.json"

# spider-dk
dataset_path="./data/spider-dk/Spider-DK.json"
tables="./data/spider-dk/tables.json"

# prepare table file for natsql
python NatSQL/table_transform.py \
    --in_file $tables \
    --out_file $tables_for_natsql \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types

# preprocess test set
python preprocessing.py \
    --mode "test" \
    --table_path $tables \
    --input_dataset_path $dataset_path \
    --output_dataset_path "./data/preprocessed_data/preprocessed_spider_dk_natsql.json" \
    --db_path "./database" \
    --target_type "natsql"

# predict probability for each schema item in the test set
python schema_item_classifier.py \
    --batch_size 32 \
    --device $device \
    --seed 42 \
    --save_path "./models/text2natsql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_spider_dk_natsql.json" \
    --output_filepath "./data/preprocessed_data/spider_dk_with_probs_natsql.json" \
    --use_contents \
    --mode "test"

# generate text2natsql test set
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/spider_dk_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_spider_dk_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "test" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"
set -e

device="0"
tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

if [ $1 = "base" ]
then
    text2natsql_model_save_path="./models/text2natsql-mt5-base-cspider/checkpoint-32448"
    text2natsql_model_bs=16
elif [ $1 = "large" ]
then
    text2natsql_model_save_path="./models/text2natsql-mt5-large-cspider/checkpoint-73691"
    text2natsql_model_bs=8
elif [ $1 = "3b" ]
then
    text2natsql_model_save_path="./models/text2natsql-mt5-xl-cspider/checkpoint-167433"
    text2natsql_model_bs=6
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

model_name="resdsql_$1_natsql"

# cspider's dev set
table_path="./data/CSpider/tables.json"
input_dataset_path="./data/CSpider/dev.json"
db_path="./database"
output="./predictions/CSpider/$model_name/pred.sql"

# prepare table file for natsql
python NatSQL/table_transform.py \
    --in_file $table_path \
    --out_file $tables_for_natsql \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types \
    --db_path $db_path

# preprocess test set
python preprocessing.py \
    --mode "test" \
    --table_path $table_path \
    --input_dataset_path $input_dataset_path \
    --output_dataset_path "./data/preprocessed_data/preprocessed_test_natsql.json" \
    --db_path $db_path \
    --target_type "natsql"

# predict probability for each schema item in the test set
python schema_item_classifier.py \
    --batch_size 32 \
    --device $device \
    --seed 42 \
    --save_path "./models/xlm_roberta_text2natsql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_test_natsql.json" \
    --output_filepath "./data/preprocessed_data/test_with_probs_natsql.json" \
    --use_contents \
    --mode "test"

# generate text2natsql test set
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/test_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_test_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "test" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# inference using the best text2natsql ckpt
python text2sql.py \
    --batch_size $text2natsql_model_bs \
    --device $device \
    --seed 42 \
    --save_path $text2natsql_model_save_path \
    --mode "eval" \
    --dev_filepath "./data/preprocessed_data/resdsql_test_natsql.json" \
    --original_dev_filepath $input_dataset_path \
    --db_path $db_path \
    --tables_for_natsql $tables_for_natsql \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --output $output
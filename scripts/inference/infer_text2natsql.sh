set -e

device="0"
tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

if [ $1 = "base" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-base/checkpoint-14352"
    text2natsql_model_bs=16
elif [ $1 = "large" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-large/checkpoint-21216"
    text2natsql_model_bs=8
elif [ $1 = "3b" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-78302"
    text2natsql_model_bs=2
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

if [ $2 = "spider" ]
then
    # spider's dev set
    dataset_path="./data/spider/dev.json"
    tables="./data/spider/tables.json"
elif [ $2 = "spider-realistic" ]
then
    # spider-realistic
    dataset_path="./data/spider-realistic/spider-realistic.json"
    tables="./data/spider/tables.json"
    if [ $1 = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
elif [ $2 = "spider-syn" ]
then
    # spider-syn
    dataset_path="./data/spider-syn/dev_syn.json"
    tables="./data/spider/tables.json"
elif [ $2 = "spider-dk" ]
then
    # spider-dk
    dataset_path="./data/spider-dk/Spider-DK.json"
    tables="./data/spider-dk/tables.json"
else
    echo "The second arg must in [spider, spider-realistic, spider-syn, spider-dk]."
    exit
fi

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
    --output_dataset_path "./data/preprocessed_data/preprocessed_test_natsql.json" \
    --db_path "./database" \
    --target_type "natsql"

# predict probability for each schema item in the test set
python schema_item_classifier.py \
    --batch_size 32 \
    --device $device \
    --seed 42 \
    --save_path "./models/text2natsql_schema_item_classifier" \
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
    --original_dev_filepath $dataset_path \
    --db_path "./database" \
    --tables_for_natsql $tables_for_natsql \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql"
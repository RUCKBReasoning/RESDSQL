set -e

device="0"

if [ $1 = "base" ]
then
    text2sql_model_save_path="./models/text2sql-t5-base/checkpoint-39312"
    text2sql_model_bs=16
elif [ $1 = "large" ]
then
    text2sql_model_save_path="./models/text2sql-t5-large/checkpoint-30576"
    text2sql_model_bs=8
elif [ $1 = "3b" ]
then
    text2sql_model_save_path="./models/text2sql-t5-3b/checkpoint-103292"
    text2sql_model_bs=2
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

# preprocess test set
python preprocessing.py \
    --mode "test" \
    --table_path $tables \
    --input_dataset_path $dataset_path \
    --output_dataset_path "./data/preprocessed_data/preprocessed_test.json" \
    --db_path "./database" \
    --target_type "sql"

# predict probability for each schema item
python schema_item_classifier.py \
    --batch_size 32 \
    --device $device \
    --seed 42 \
    --save_path "./models/text2sql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_test.json" \
    --output_filepath "./data/preprocessed_data/test_with_probs.json" \
    --use_contents \
    --add_fk_info \
    --mode "test"

# generate text2sql test set
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/test_with_probs.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_test.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "test" \
    --use_contents \
    --add_fk_info \
    --output_skeleton \
    --target_type "sql"

# inference using the best text2sql ckpt
python text2sql.py \
    --batch_size $text2sql_model_bs \
    --device $device \
    --seed 42 \
    --save_path $text2sql_model_save_path \
    --mode "eval" \
    --dev_filepath "./data/preprocessed_data/resdsql_test.json" \
    --original_dev_filepath $dataset_path \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "sql"
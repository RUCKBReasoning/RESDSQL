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
    text2sql_model_bs=6
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

model_name="resdsql_$1"

if [ $2 = "spider" ]
then
    # spider's dev set
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider/dev.json"
    db_path="./database"
    output="./predictions/Spider-dev/$model_name/pred.sql"
elif [ $2 = "spider-realistic" ]
then
    # spider-realistic
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider-realistic/spider-realistic.json"
    db_path="./database"
    output="./predictions/spider-realistic/$model_name/pred.sql"
    if [ $1 = "3b" ]
    then
        text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-61642"
    fi
elif [ $2 = "spider-syn" ]
then
    # spider-syn
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider-syn/dev_syn.json"
    db_path="./database"
    output="./predictions/spider-syn/$model_name/pred.sql"
elif [ $2 = "spider-dk" ]
then
    # spider-dk
    table_path="./data/spider-dk/tables.json"
    input_dataset_path="./data/spider-dk/Spider-DK.json"
    db_path="./database"
    output="./predictions/spider-dk/$model_name/pred.sql"
elif [ $2 = "DB_DBcontent_equivalence" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/tables_post_perturbation.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/database_post_perturbation"
    output="./predictions/DB_DBcontent_equivalence/$model_name/pred.sql"
elif [ $2 = "DB_schema_abbreviation" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/DB_schema_abbreviation/tables_post_perturbation.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/DB_schema_abbreviation/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/DB_schema_abbreviation/database_post_perturbation"
    output="./predictions/DB_schema_abbreviation/$model_name/pred.sql"
elif [ $2 = "DB_schema_synonym" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/DB_schema_synonym/tables_post_perturbation.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/DB_schema_synonym/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/DB_schema_synonym/database_post_perturbation"
    output="./predictions/DB_schema_synonym/$model_name/pred.sql"
elif [ $2 = "NLQ_column_attribute" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_attribute/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_attribute/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_attribute/databases"
    output="./predictions/NLQ_column_attribute/$model_name/pred.sql"
elif [ $2 = "NLQ_column_carrier" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_carrier/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_carrier/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_carrier/databases"
    output="./predictions/NLQ_column_carrier/$model_name/pred.sql"
elif [ $2 = "NLQ_column_synonym" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_synonym/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_synonym/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_synonym/databases"
    output="./predictions/NLQ_column_synonym/$model_name/pred.sql"
elif [ $2 = "NLQ_column_value" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_value/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_value/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_column_value/databases"
    output="./predictions/NLQ_column_value/$model_name/pred.sql"
elif [ $2 = "NLQ_keyword_carrier" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_keyword_carrier/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_keyword_carrier/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_keyword_carrier/databases"
    output="./predictions/NLQ_keyword_carrier/$model_name/pred.sql"
elif [ $2 = "NLQ_keyword_synonym" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_keyword_synonym/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_keyword_synonym/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_keyword_synonym/databases"
    output="./predictions/NLQ_keyword_synonym/$model_name/pred.sql"
elif [ $2 = "NLQ_multitype" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_multitype/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_multitype/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_multitype/databases"
    output="./predictions/NLQ_multitype/$model_name/pred.sql"
elif [ $2 = "NLQ_others" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_others/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_others/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_others/databases"
    output="./predictions/NLQ_others/$model_name/pred.sql"
elif [ $2 = "NLQ_value_synonym" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_value_synonym/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_value_synonym/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/NLQ_value_synonym/databases"
    output="./predictions/NLQ_value_synonym/$model_name/pred.sql"
elif [ $2 = "SQL_comparison" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/SQL_comparison/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/SQL_comparison/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/SQL_comparison/databases"
    output="./predictions/SQL_comparison/$model_name/pred.sql"
elif [ $2 = "SQL_DB_number" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/SQL_DB_number/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/SQL_DB_number/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/SQL_DB_number/databases"
    output="./predictions/SQL_DB_number/$model_name/pred.sql"
elif [ $2 = "SQL_DB_text" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/SQL_DB_text/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/SQL_DB_text/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/SQL_DB_text/databases"
    output="./predictions/SQL_DB_text/$model_name/pred.sql"
elif [ $2 = "SQL_NonDB_number" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/SQL_NonDB_number/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/SQL_NonDB_number/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/SQL_NonDB_number/databases"
    output="./predictions/SQL_NonDB_number/$model_name/pred.sql"
elif [ $2 = "SQL_sort_order" ]
then
    table_path="./data/diagnostic-robustness-text-to-sql/data/SQL_sort_order/tables.json"
    input_dataset_path="./data/diagnostic-robustness-text-to-sql/data/SQL_sort_order/questions_post_perturbation.json"
    db_path="./data/diagnostic-robustness-text-to-sql/data/SQL_sort_order/databases"
    output="./predictions/SQL_sort_order/$model_name/pred.sql"
else
    echo "The second arg must in [spider, spider-realistic, spider-syn, spider-dk, DB_schema_synonym, DB_schema_abbreviation, DB_DBcontent_equivalence, NLQ_keyword_synonym, NLQ_keyword_carrier, NLQ_column_synonym, NLQ_column_carrier, NLQ_column_attribute, NLQ_column_value, NLQ_value_synonym, NLQ_multitype, NLQ_others, SQL_comparison, SQL_sort_order, SQL_NonDB_number, SQL_DB_text, SQL_DB_number]."
    exit
fi

# preprocess test set
python preprocessing.py \
    --mode "test" \
    --table_path $table_path \
    --input_dataset_path $input_dataset_path \
    --output_dataset_path "./data/preprocessed_data/preprocessed_test.json" \
    --db_path $db_path \
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
    --original_dev_filepath $input_dataset_path \
    --db_path $db_path \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "sql" \
    --output $output
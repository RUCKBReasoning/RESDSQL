set -e

# # setup the environment
# pip install -r requirements.txt
# python nltk_downloader.py

# arg values
device="0"
mode="eval"
dataset_path="data/dev.json"
table_path="data/tables.json"
database_path="database"

# preprocessing
python preprocessing.py \
    --use_contents \
    --table_path $table_path \
    --mode $mode \
    --input_dataset_path $dataset_path \
    --output_dataset_path preprocessed_dataset.json \
    --db_path $database_path

# run ranker model
python ranker.py \
    --batch_size 32 \
    --device $device \
    --save_path ranker_model \
    --test_dataset_path preprocessed_dataset.json \
    --output_dataset_path ranked_dataset.json \
    --use_contents \
    --mode $mode

# run text2sql model (xl scale)
python text2sql.py \
    --batch_size 4 \
    --device $device \
    --save_path text2sql_xl \
    --use_contents \
    --add_fk_info \
    --output_sql_skeleton \
    --mode $mode \
    --ranked_test_dataset_filepath ranked_dataset.json \
    --original_dev_dataset_filepath $dataset_path \
    --db_path $database_path \
    --checkpoint_type EM \
    --num_beams 8 \
    --num_return_sequences 8
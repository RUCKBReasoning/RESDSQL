set -e

# preprocess train_spider dataset
python preprocessing.py \
    --mode "train" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/train_spider.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_train_spider.json" \
    --db_path "./database" \
    --target_type "sql"

# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/dev.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev.json" \
    --db_path "./database"\
    --target_type "sql"
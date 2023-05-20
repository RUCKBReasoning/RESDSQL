set -e

# preprocess train_cspider dataset
python preprocessing.py \
    --mode "train" \
    --table_path "./data/CSpider/tables.json" \
    --input_dataset_path "./data/CSpider/train.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_train_cspider.json" \
    --db_path "./database" \
    --target_type "sql"

# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/CSpider/tables.json" \
    --input_dataset_path "./data/CSpider/dev.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider.json" \
    --db_path "./database" \
    --target_type "sql"
set -e

# preprocess train_cspider dataset
python preprocessing.py \
    --mode "train" \
    --table_path "./data/CSpider/tables.json" \
    --input_dataset_path "./data/CSpider/train_cspider.json" \
    --natsql_dataset_path "./NatSQL/NatSQLv1_6/train_cspider-natsql.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_train_cspider_natsql.json" \
    --db_path "./database" \
    --target_type "natsql"

# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/CSpider/tables.json" \
    --input_dataset_path "./data/CSpider/dev.json" \
    --natsql_dataset_path "./NatSQL/NatSQLv1_6/dev_cspider-natsql.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_natsql.json" \
    --db_path "./database" \
    --target_type "natsql"

# preprocess tables.json for natsql
python NatSQL/table_transform.py \
    --in_file "./data/CSpider/tables.json" \
    --out_file "./data/preprocessed_data/cspider_tables_for_natsql.json" \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types
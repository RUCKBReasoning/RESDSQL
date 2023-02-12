if [ ! -f "data/dev.json" ];then
  echo "Can not find the data/dev.json"
  exit
fi

if [ ! -f "data/train_spider.json" ];then
  echo "Can not find the data/train_spider.json"
  exit
fi

if [ ! -f "data/tables.json" ];then
  echo "Can not find the data/tables.json"
  exit
fi

if [ ! -d "data/database" ];then
  echo "Can not find the data/database"
  exit
fi

if [ ! -d "data/database/academic" ];then
  echo "Can not find the data/database"
  exit
fi


python table_transform.py --in_file data/tables.json --out_file NatSQLv1_6/tables_for_natsql.json --correct_col_type --remove_start_table  --analyse_same_column --table_transform --correct_primary_keys --use_extra_col_types
python table_transform.py --in_file data/tables.json --out_file NatSQLv1_6/tables.json --correct_col_type  --remove_start_table --seperate_col_name --analyse_same_column --add_debug_col --use_extra_col_types
# python punkt.py
python generate_spider_examples_with_natsql.py
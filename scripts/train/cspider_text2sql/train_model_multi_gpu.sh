# train text2natsql-mt5-xl-cspider model
python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 text2sql.py \
    --batch_size 4 \
    --gradient_descent_step 4 \
    --device "1,2,3" \
    --learning_rate 5e-5 \
    --epochs 512 \
    --seed 42 \
    --save_path "./models/myTrain/text2sql-mt5-large-medical-cspider-multigpu-better-data" \
    --tensorboard_save_path "./tensorboard_log/text2sql-mt5-large-medical-spider-multigpu-better-data" \
    --model_name_or_path "./models/mt5-large-raw" \
    --dev_filepath "./data/Medical/preprocessed_data/train/resdsql_dev_medical_cspider_natsql.json" \
    --use_adafactor \
    --mode "train" \
    --db_path "./data/Medical/database" \
    --train_filepath "./data/Medical/preprocessed_data/train/resdsql_train_medical_cspider_natsql.json" \
    --original_dev_filepath "./data/Medical/dev_medical_cspider.json"
#!/bin/bash

dev_or_train=$1
natsql_or_natsqlg=$2

if [ "$dev_or_train" = "dev" ];
then
if [ ! -f "dev-preprocessed.json" ];then
# You can add a --keep_or argument following the setence_split.py and pattern_generation.py to un-change the original Spider question.
python setence_split.py --in_file NatSQLv1_6/dev.json  --out_file dev-ss.json  
python pattern_generation.py --in_file dev-ss.json  --out_file dev-preprocessed.json 
rm dev-ss.json 
fi
Path="dev-preprocessed.json"
elif [ "$dev_or_train" = "train" ];
then
if [ ! -f "train_spider-preprocessed.json" ];then
# You can add a --keep_or argument following the setence_split.py and pattern_generation.py to un-change the original Spider question.
python setence_split.py --in_file NatSQLv1_6/train_spider.json  --out_file train_spider-ss.json  
python pattern_generation.py --in_file train_spider-ss.json  --out_file train_spider-preprocessed.json 
rm train_spider-ss.json 
fi
Path="train_spider-preprocessed.json"
else
echo "The first paprmeter must be dev or train."
exit
fi


if [ "$natsql_or_natsqlg" = "natsql" ];
then
python run.py --natsql_file $Path --remove_groupby_from_natsql --test_executable_natsql 1>results.sql
elif [ "$natsql_or_natsqlg" = "natsqlg" ];
then
python run.py --natsql_file $Path --test_executable_natsql 1>results.sql
else
echo "The second paprmeter must be natsql or natsqlg."
exit
fi
